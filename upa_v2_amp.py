# coding=UTF-8<code>

import argparse
import os
import os.path as osp
import datetime
import torch
import numpy as np
import random
from utils.tools import print_args, image_train
from utils.str2bool import str2bool
import json
from trainer.engine import Upa

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home',
                        choices=['VISDA-C', 'office', 'office-home', 'office-caltech', 'domainnet126'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='res/ckps/source')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])

    parser.add_argument('--issave', type=str2bool, default=False)
    
    parser.add_argument('--run_all', type=str2bool, default=True, help='whether to run all target for source')
    parser.add_argument('--sel_cls', type=str2bool, default=True, help='whether to select samples for cls loss')
    parser.add_argument('--balance_class', type=str2bool, default=True,
                        help='whether to balance class in pair_selection')
    parser.add_argument('--knn_times', type=int, default=2, help='how many times of knn is conducted')

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
    parser.add_argument('--folder', type=str, default='../DATASETS/')
    args = parser.parse_args()
    args.append_root = None
    folder = args.folder

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
        args.run_all = False
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
    if args.dset == 'domainnet126':
        names = ['clipart', 'painting', 'real', 'sketch']
        args.class_num = 126
        args.append_root = f'{folder}/domainnet126/'

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    
    def tem_run():
        startTime = datetime.datetime.now()
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        args.savename = f'task_{args.name}_{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}_par_n_ent_{args.par_noisy_ent}_par_su_cl_{args.par_su_cl}_tau2_{args.su_cl_t}_kval_{args.k_val}_selr_{args.sel_ratio}_knnt_{args.knn_times}'
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        print(f'args:{args}')
        args.out_file.flush()

        upaBuilder = Upa(args)
        acc_final = upaBuilder.start_train()
        
        #set time stamp
        endTime = datetime.datetime.now()
        dua_time = (endTime-startTime).seconds
        startTime = endTime
        log_str = f'Time consumed:{dua_time}'
        print(log_str)
        args.out_file.write(log_str+'\n' + '-'*60 + '\n')
        args.out_file.flush()
        return acc_final
    
    res_dict = {}
    ##save scripts
    args.output_dir = osp.join(args.output, args.da, args.dset)
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    if args.run_all:
        for i in range(len(names)):
            for j in range(len(names)):
                if j == i:
                    continue
                args.s = i
                args.t = j
                acc = tem_run()
                res_dict[names[args.s][0].upper()+names[args.t][0].upper()] = acc
        def cal_avg_acc(dic):
            sum_res = 0
            n = 0
            for k,v in dic.items():
                sum_res+=v
                n+=1
            return round(sum_res/n,1)
        log_str = 'final result:'+'\n'+json.dumps(res_dict)
        args.out_file.write(log_str)
        log_str = f'Avg acc: {cal_avg_acc(res_dict)}%'
        args.out_file.write(log_str)
        args.out_file.flush()
    else:
        acc = tem_run()
        

    
    