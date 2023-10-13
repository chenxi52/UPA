# coding=UTF-8<code>
# version of cvpr2023

import argparse
import os
import os.path as osp
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import datetime
import torch
import numpy as np
import random
from model import network
import torch.nn.functional as F
from loaders.target_loader import data_load
from loaders.data_list import Pseudo_dataset
from torch.utils.data import DataLoader
from utils.adjust_par import op_copy
from utils.tools import print_args, image_train
from utils.str2bool import str2bool
from utils.adjust_par import cosine_warmup
from utils import loss as Loss
from utils.utils_noise import pair_selection_v1, pair_selection
import json
from torch.cuda.amp import autocast
from model.network import ResBase, VGGBase, feat_bootleneck


def forward():
    # encoder.eval()
    netF.eval()
    netB.eval()
    netC.eval()
    _ = obtain_label()
    if args.dset == 'VISDA-C':
        acc_s_te, acc_list = cal_acc(True)
        log_str = f'Task: {args.name}, Accuracy = {acc_s_te:.2f}; \n {acc_list}'
    else:
        acc_s_te = cal_acc(False)
        log_str = f'Task: {args.name}, Accuracy = {acc_s_te:.2f} ;'
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')
        

def cal_acc(flag=False):
    start_test = True
    with torch.no_grad():
        iter_train = iter(loader['test'])
        for i in range(len(loader['test'])):
            data = iter_train.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat([all_output, outputs.float().cpu()], 0)
                all_label = torch.cat([all_label, labels.float()], 0)
    
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()#每类正确率的均值
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc  # acc是单类的正确率 aacc是acc的均值
    else:
        return accuracy * 100

def obtain_label(return_dist=False):
    start_test = True
    with torch.no_grad():
        iter_train = iter(loader['test'])
        for _ in range(len(loader['test'])):
            data = iter_train.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat([all_fea, feas.float().cpu()], 0)
                all_output = torch.cat([all_output, outputs.float().cpu()], 0)
                all_label = torch.cat([all_label, labels.float()], 0)

    all_output = nn.Softmax(dim=1)(all_output)  # N*C
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat([all_fea, torch.ones(all_fea.size(0), 1)], 1)  # N*(f+1),这里加了一列全1的
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()  # n*(f+1)
        all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()  # n*k
    initc = aff.transpose().dot(all_fea)  # k*n*(n*(f+1))=k*(f+1)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])  # 扩充了一维分母变为k*1----(k*(f+1))/(k*1)=k*(f+1)(即每一行是一类的d维中心）
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > 0)
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], args.distance)  # cdist(N*f+1,k*f+1)-->N*k+1
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]  # N*K
        initc = aff.transpose().dot(all_fea)  # K*(f+1)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]  # (N,)

    min_dist = dd.min(axis=1)  # 样本距离自己的聚类中心的距离
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = f'Task:{args.name}   Accuracy = {accuracy * 100:.2f}% -> {acc * 100:.2f}%'
    # accuracy:目前的模型预测成功率；
    # acc：pseudo label和true label相似度

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')

    if return_dist:
        return pred_label.astype('int'), min_dist
    else:
        return pred_label.astype('int'), torch.from_numpy(all_fea[:, :-1]).cuda(), initc[:, :-1], all_label.float().cuda(), all_output.cuda()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home',
                        choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='res/ckps/source-30')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])

    parser.add_argument('--issave', type=str2bool, default=False)
    
    parser.add_argument('--run_all', type=str2bool, default=True, help='whether to run all target for source')
    parser.add_argument('--sel_cls', type=str2bool, default=True, help='whether to select samples for cls loss')
    parser.add_argument('--balance_class', type=str2bool, default=True,
                        help='whether to balance class in pair_selection')
    parser.add_argument('--knn_times', type=int, default=2, help='how many times of knn is conducted')
    # parser.add_argument('--test_sel_acc',type=str2bool,default=False,help='whether to calculate selection accuacy to its real labels')

    # weight of losses
    parser.add_argument('--par_cls', type=float, default=0.3)
    parser.add_argument('--par_ent', type=float, default=1.0)
    parser.add_argument('--par_noisy_cls', type=float, default=0.3)
    parser.add_argument('--par_noisy_ent', type=float, default=1.0)
    parser.add_argument('--par_su_cl', type=float, default=1.)

    # contrastive learning params
    parser.add_argument('--su_cl_t', type=float, default=5., help='tem for supervised contrastive loss')

    # pseudo-labeling params
    parser.add_argument('--k_val', type=int, default=30, help='knn neighbors number')
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--sel_ratio', type=float, default=0, help='sel_ratio for clean_samples')
    parser.add_argument('--cos_t', type=float, default=5, help='tem for knn prob estimation')

    # network params
    parser.add_argument('--net', type=str, default='resnet50',
                        help="alexnet, vgg16, resnet50, resnet101,vit")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn", "bn_drop"])

    # data augmentation
    parser.add_argument('--aug', type=str, default='mocov2', help='strong augmentation type')

    # train schedule
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--scheduler_warmup_epochs', type=int, default=1)

    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
        args.warmup_epochs = 1
        args.lr = 1e-3
        args.net = 'resnet101'
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    folder = 'data/datasets/'
    
    args.t_dset_path = folder + args.dset + '/' + names[args.t][0] + '_list.txt'
    args.test_dset_path = folder + args.dset + '/' + names[args.t][0] + '_list.txt'

    args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
    args.name = names[args.s][0].upper() + names[args.t][0].upper()

    if args.net[0:3] == 'res':
        netF = ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = VGGBase(vgg_name=args.net).cuda()
    
    netB = feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                bottleneck_dim=args.bottleneck).cuda()
    modelpath = osp.join(args.output_dir_src, 'source_F.pt')
    netF.load_state_dict(torch.load(modelpath))
    modelpath = osp.join(args.output_dir_src, 'source_B.pt')
    netB.load_state_dict(torch.load(modelpath))

    netC = network.feat_classifier(type=args.layer, class_num=args.class_num,
                                        bottleneck_dim=args.bottleneck).cuda()
    modelpath = os.path.join(args.output_dir_src, 'source_C.pt')
    netC.load_state_dict(torch.load(modelpath))

    # encoder = encoder.cuda()
    netC = netC.cuda()
    loader, dsets = data_load(args)

    ##save scripts
    args.output_dir = osp.join(args.output, args.da, args.dset)
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)
        
    args.savename = f'{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}_par_n_ent_{args.par_noisy_ent}_par_su_cl_{args.par_su_cl}_tau2_{args.su_cl_t}_kval_{args.k_val}_selr_{args.sel_ratio}_knnt_{args.knn_times}'
    args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    print(f'args:{args}')
    args.out_file.flush()
    
    startTime = datetime.datetime.now()
    acc_final = forward()
    
    #set time stamp
    endTime = datetime.datetime.now()
    dua_time = (endTime-startTime).seconds
    startTime = endTime
    log_str = f'Time consumed:{dua_time}'
    print(log_str)
    args.out_file.write(log_str+'\n' + '-'*60 + '\n')
    args.out_file.flush()
 
        

    
    