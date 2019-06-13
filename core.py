
import torch
import torch.distributed as dist
from apex import amp
from common import get_world_size, is_main_process
from utils import make_data_sampler


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def reduce_loss(loss):
    world_size = get_world_size()
    if world_size < 2:
        return loss
    with torch.no_grad():
        all_loss = []
        dist.reduce(all_loss, dst=0)
        if dist.get_rank() == 0:
            all_loss /= world_size
    return all_loss

def do_train(
    args,
    model,
    optimizer,
    device
):
    model.train()

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
        shuffle=(not args.distributed),
        num_workers=4,
        pin_memory=True,
    )
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)
            
        for i, (images, targets) in enumerate(data_loader):
            images = images.to(device)
            targets = targets.to(device)

            loss = model(images, targets)
            # 显示loss
            loss_reduced = reduce_loss(loss)
            
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            if is_main_process() and i % args.print_freq == 0:
                print('{} epoch, {} iter, loss: {}'.format(epoch, i, loss_reduced))


