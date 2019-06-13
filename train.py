import os
import argparse
import torch
from apex import amp

from comm import synchronize
from EfficientNet import EfficientNet
from core import do_train

def train(args, local_rank, distributed):
    model = EfficientNet.from_name(args.arch)
    device = torch.device("cuda")
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    amp_opt_level = 'O0'
    if args.float16:
        amp_opt_level = 'O1'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            # 更新BN参数，comment下面选项
            # broadcast_buffers=False,
        )

    do_train(
        args,
        model,
        optimizer,
        device,
    )


def main():
    parser = argparse.ArgumentParser(description="Distributed training")
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    # TODO: 增加模型训练的设置
    parser.add_argument('--arch', type=str, default='efficientnet-b0')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--float16', type=bool, default=False)
    parser.add_argument('--data', type=str, default='data/')
    parser.add_argument('--batch-size', type=int, default=64, help="Images per gpu")
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--print-freq', type=int, default=100)

    args = parser.parse_args()
    num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
        )
        synchronize()

    # 训练
    model = train(args, args.local_rank, args.distributed)
    if not args.skip_test:
        run_test(args, model, args.distributed)
    


if __name__ == "__main__":
    main()
