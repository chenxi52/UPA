import numpy as np


def cosine_keep(lr_min, lr_max, T_cur, T_max):
    if T_cur < T_max:
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos((T_cur + T_max) / (T_max) * math.pi))
    else:
        lr = lr_max
    return lr


def cosine_warmup(optimizer, iter, max_iters, lr_min=0, warmup_iters=0):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if iter < warmup_iters:
        lr_decay = iter / warmup_iters

        for param_group in optimizer.param_groups:
            # param_group['lr'] =(1e-2) * param_group['lr0']+ param_group['lr0']*(1-1e-2) * lr_decay
            param_group['lr'] = param_group['lr0'] * lr_decay

    else:
        lr_decay = 0.5 * (1. + math.cos(math.pi * (iter - warmup_iters) / (max_iters - warmup_iters)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_min + (param_group['lr0'] - lr_min) * lr_decay


def keep_cosine(optimizer, iter, max_iters, lr_min=0, keep_iters=0):
    if iter < keep_iters:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr0']
    else:
        lr_decay = 0.5 * (1. + math.cos(math.pi * (iter - keep_iters) / (max_iters - keep_iters)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_min + (param_group['lr0'] - lr_min) * lr_decay


import math


def adjust_lr(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def step_lr(epoch, lr_decay_every, lr_decay_factor, optimizer):
    """Handles step decay of learning rate."""
    factor = np.power(lr_decay_factor, np.floor((epoch - 1) / lr_decay_every))
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * factor
    print('Set lr to ', param_group['lr0'] * factor)
    return optimizer
