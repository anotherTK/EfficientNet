import torch


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler
