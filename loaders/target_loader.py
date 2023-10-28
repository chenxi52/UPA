import torch.utils.data

from loaders.data_list import ImageList, ImageList_idx, Two_ImageList_idx, ImageList_multi_transform
from utils.tools import *
from torch.utils.data import DataLoader


def data_load(args, distributed=False):
    strong_aug = {
                  'mocov2': mocov2(),
                  }
    ## prepare data
    dsets = {}
    dset_loaders = {}
    samplers = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                    new_tar.append(line)
                else:
                    # only be forwarded when da=='oda'
                    # class that not exist in both source and target domains are concluded to one class.
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train(), append_root=args.append_root)
    if distributed:
        samplers['target'] = torch.utils.data.distributed.DistributedSampler(dsets["target"], shuffle=True)
    else:
        samplers['target'] = None
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, num_workers=args.worker,
                                        shuffle=(distributed is False),
                                        drop_last=False, sampler=samplers['target'], pin_memory=True)

    dsets["test"] = ImageList_idx(txt_test, transform=image_test(), append_root=args.append_root)
    ###test loader绝对不能shuffle ，因为obtain label要保持index不变
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs, shuffle=False, num_workers=args.worker,
                                      drop_last=False, sampler=None, pin_memory=True)
    if distributed:
        samplers['test'] = torch.utils.data.DistributedSampler(dsets['test'], shuffle=False)
        ###test loader绝对不能shuffle ，因为obtain label要保持index不变
        dset_loaders["d_test"] = DataLoader(dsets["test"], batch_size=train_bs, num_workers=args.worker,
                                            drop_last=False, sampler=samplers['test'], pin_memory=True)

    dsets["two_train"] = Two_ImageList_idx(txt_tar, transform1=image_train(), transform2=strong_aug[args.aug], append_root=args.append_root)
    if distributed:
        samplers['two_train'] = torch.utils.data.distributed.DistributedSampler(dsets["two_train"], shuffle=True)
    else:
        samplers['two_train'] = None
    dset_loaders["two_train"] = DataLoader(dsets["two_train"], batch_size=train_bs, num_workers=args.worker,
                                           shuffle=True,
                                           drop_last=False, sampler=samplers['two_train'], pin_memory=True)
    # dsets["two_train_test"] = ImageList_multi_transform(txt_tar, transform=[image_train(), strong_aug[args.aug], image_test()])
    # dset_loaders["two_train_test"] = DataLoader(dsets["two_train_test"], batch_size=train_bs, num_workers=args.worker,
    #                                             shuffle=True, drop_last=False, sampler=None, pin_memory=True)
    # dset_loaders['queue_two_train'] = DataLoader(dsets["two_train"], batch_size=train_bs, num_workers=)
    if distributed:
        return dset_loaders, samplers
    else:
        return dset_loaders, dsets


def data_load_mda(args):
    strong_aug = {'image_train': image_train(),
                  'mocov2': mocov2(),
                  'randaug': randaugment(N=args.rand_N, M=args.rand_M),
                  'randaug_gray': randaug_gray(N=args.rand_N, M=args.rand_M),
                  'randaug2': randaugment2(N=args.rand_N, M=args.rand_M),
                  'randaug_gray2': randaug_gray2(N=args.rand_N, M=args.rand_M),
                  'randaug_resizecrop': randaug_resizecrop(N=args.rand_N, M=args.rand_M),
                  'randaug_jitter': randaug_jitter(N=args.rand_N, M=args.rand_M),

                  }
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = []
    for i in range(len(args.t_dset_path)):
        tmp = open(args.t_dset_path[i]).readlines()
        txt_tar.extend(tmp)
    txt_test = txt_tar.copy()


    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, num_workers=args.worker,
                                        shuffle=True, drop_last=False, pin_memory=True)

    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    ###test loader绝对不能shuffle ，因为obtain label要保持index不变
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs, shuffle=False, num_workers=args.worker,
                                      drop_last=False, pin_memory=True)

    dsets["two_train"] = Two_ImageList_idx(txt_tar, transform1=image_train(), transform2=strong_aug[args.aug])
    dset_loaders["two_train"] = DataLoader(dsets["two_train"], batch_size=train_bs, num_workers=args.worker,
                                           shuffle=True, drop_last=False, pin_memory=True)

    return dset_loaders, dsets


def two_dataload(args):
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    dset = Two_ImageList_idx(txt_tar, transform1=image_train(), transform2=image_test())
    dset_loader = DataLoader(dset, batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=True)
    return dset_loader


import math


# 来源：https://github.com/huggingface/transformers/blob/447808c85f0e6d6b0aeeb07214942bf1e578f9d2/src/transformers/trainer_pt_utils.py
class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples: (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples


# 合并结果的函数
# 1. all_gather，将各个进程中的同一份数据合并到一起。
#   和all_reduce不同的是，all_reduce是平均，而这里是合并。
# 2. 要注意的是，函数的最后会裁剪掉后面额外长度的部分，这是之前的SequentialDistributedSampler添加的。
# 3. 这个函数要求，输入tensor在各个进程中的大小是一模一样的。
def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]

def ori_data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["target"] = ImageList(txt_tar, transform=image_test())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders