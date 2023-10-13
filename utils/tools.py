from torchvision import transforms
import numpy as np
import torch
from PIL import ImageFilter
import random

def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # normalize = transforms.Normalize(mean=[0.5,0.5,0.5],
        #                                   std=[0.5,0.5,0.5])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
def image_normalize(resize_size=224, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
        #                                  std=[0.5, 0.5, 0.5])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor(),
        normalize
    ])
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def mocov2(resize_size=256,crop_size=224,alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
        #                                  std=[0.5, 0.5, 0.5])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    augmentation =transforms.Compose ([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    return augmentation



def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def mixup_data(x1, x2,alpha=0.3, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x1.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x1 + (1 - lam) * x2[index, :]
    return mixed_x

def mixup_data_uneven(x,y,alpha=0.3):
    '''
    :param alpha: mixup参数
    :param com_ind: 和low_prob sample mix的high_sample样本在自身集合中的索引
    :return: 一个batch中的样本和其中高概率样本的混合
    '''

    if alpha>0:
        lam=np.random.beta(alpha,alpha)
    else:
        lam=1
    batch_size=x.size(0)
    index = torch.randperm(batch_size).cuda()
    lam = max(lam, 1 - lam)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a,y_b=y,y[index]
    return mixed_x,y_a,y_b,index,lam

#使用多进程时，每个进程有自己计算得到的Loss，在进行数据记录时，对不同进程上的Loss取平均（也就是 map-reduce 的做法
from torch import distributed
def reduce_tensor(tensor):
    rt = tensor.clone()
    distributed.all_reduce(rt, op=distributed.reduce_op.SUM)
    rt /= distributed.get_world_size()
    return rt


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def to_onehot(label, num_classes):
    identity = torch.eye(num_classes).cuda()
    onehot = torch.index_select(identity, 0, label)
    return onehot
def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)

def set_bn_train(model):
    def set_bn_train_helper(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.train()

    model.eval()
    model.apply(set_bn_train_helper)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]