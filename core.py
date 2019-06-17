
import os
import torch
import torch.distributed as dist
import torchvision
from apex import amp
from comm import get_world_size, is_main_process
from utils import make_data_sampler, MetricLogger
import logging
import time
import datetime
from tqdm import tqdm
from comm import synchronize, all_gather, is_main_process


def reduce_loss(loss):
    world_size = get_world_size()
    if world_size < 2:
        return loss
    with torch.no_grad():
        all_loss = loss.clone()
        dist.reduce(all_loss, dst=0)
        if dist.get_rank() == 0:
            all_loss /= world_size
    return all_loss

def do_train(
    args,
    model,
    optimizer,
    scheduler,
    device
):
    # 日志
    logger = logging.getLogger("efficientnet.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    # 数据
    traindir = os.path.join(args.data, 'train')
    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = make_data_sampler(train_dataset, True, args.distributed)
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )

    start_training_time = time.time()
    end = time.time()

    model.train()
    max_iter = len(data_loader)
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        scheduler.step()
        for iteration, (images, targets) in enumerate(data_loader):
            data_time = time.time() - end

            images = images.to(device)
            targets = targets.to(device)

            loss = model(images, targets)
            # 显示loss
            loss_reduced = reduce_loss(loss)
            meters.update(loss=loss_reduced)
            
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            # 记录时间参数
            batch_time = time.time() - end
            end = time.time()
            meters.update(
                time=batch_time,
                data=data_time
            )
            eta_seconds = meters.time.global_avg * (max_iter * (args.epochs - epoch -1 ) - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            # 显示训练状态
            if iteration % args.print_freq == 0:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "epoch: {epoch}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem (MB): {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        epoch=epoch,
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )

        # 保存checkpoint
        if is_main_process() and ((epoch + 1) % args.ckpt_freq == 0 or (epoch + 1) == args.epochs):
            ckpt = {}
            ckpt['model'] = model.state_dict()
            ckpt['optimizer'] = optimizer.state_dict()
            save_file = os.path.join(args.output_dir, "efficientnet-epoch-{}.pth".format(epoch))
            torch.save(data, save_file)
        
        # validate
        do_eval(args, model, args.distributed)

    # 总体训练时长
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter * args.epochs)
        )
    )


def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_list = []
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            outputs = model(images.to(device))
            if timer:
                # CPU和GPU同步
                torch.cuda.synchronize()
                timer.toc()
            outputs = outputs.to(cpu_device)
        results_list.append((outputs, targets))
    model.train()
    return results_list


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    """return the data_list of each rank
    """
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    targets = []
    outputs = []
    for p_per_gpu in all_predictions:
        for p in p_per_gpu:
            outputs.append(p[0])
            targets.append(p[1])
    outputs = torch.cat(outputs)
    targets = torch.cat(targets)
    return (outputs, targets)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def do_eval(args, model, distributed):
    device = torch.device("cuda")
    logger = logging.getLogger("efficientnet.inference")
    logger.info("Start inference")
    # 数据
    valdir = os.path.join(args.data, 'val')
    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    val_dataset = torchvision.datasets.ImageFolder(
        valdir,
        torchvision.transforms.Compose([
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.ToTensor(),
            normalize,
        ]))

    val_sampler = make_data_sampler(val_dataset, False, distributed)
    data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )

    predictions = compute_on_dataset(model, data_loader, device)
    # 每个进程同步
    synchronize()
    # 汇总每个进程的inference结果
    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    
    # 计算top1和top5
    if is_main_process():
        acc1, acc5 = accuracy(predictions[0], predictions[1], topk=(1, 5))
        logger.info("accuracy: top-1/ {}, top5/ {}".format(acc1, acc5))

