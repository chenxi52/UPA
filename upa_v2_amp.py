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
from utils.adjust_par import op_copy, cosine_warmup
from utils.tools import print_args, image_train
from utils.str2bool import str2bool
from utils import loss as Loss
from utils.utils_noise import pair_selection_v1, pair_selection
import json
from torch.cuda.amp import autocast

class Upa(object):
    def __init__(self, args):
        super(Upa, self).__init__()

        self.encoder = network.encoder(args)
        self.encoder.load_model()
        self.netC = network.feat_classifier(type=args.layer, class_num=args.class_num,
                                            bottleneck_dim=args.bottleneck).cuda()
        modelpath = os.path.join(args.output_dir_src, 'source_C.pt')
        self.netC.load_state_dict(torch.load(modelpath))

        self.encoder = self.encoder.cuda()
        self.netC = self.netC.cuda()
        self.loader, self.dsets = data_load(args)
        self.max_iters = len(self.loader['two_train']) * args.max_epoch
        self.scaler = torch.cuda.amp.GradScaler()

    def train_uns(self, epoch, adjust_learning_rate):
        for batchidx, (inputs, _, _, tar_idx) in enumerate(self.loader['two_train']):
            self.optimizer.zero_grad()
            if inputs.size(0) < 2:
                continue
            inputs = inputs.cuda()

            adjust_learning_rate(self.optimizer, (epoch - 1) * len(self.loader['two_train']) + batchidx + 1,
                                 self.max_iters,
                                 warmup_iters=args.scheduler_warmup_epochs * len(self.loader['two_train']))
            with autocast():
                features = self.encoder(inputs)
                outputs = self.netC(features)
                classifier_loss = torch.tensor(0.).cuda()

                softmax_out = nn.Softmax(dim=1)(outputs)  # (N,k)
                entropy_loss = torch.mean(Loss.Entropy(softmax_out))
                im_loss = entropy_loss * args.par_ent
                msoftmax = softmax_out.mean(dim=0)  # 降维（K，）
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                im_loss -= gentropy_loss * args.par_ent
                classifier_loss += im_loss

            self.scaler.scale(classifier_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
        return classifier_loss.item()


    def train_su_cl(self, epoch, trainloader, trainSelloader, mem_label, selected_pairs, adjust_learning_rate):
        if trainSelloader:
            # 这些是选出来的clean pseudo-labeled samples
            train_sel_iter = iter(trainSelloader)

        for batchidx, (inputs, m_inputs, _, index) in enumerate(trainloader):
            self.optimizer.zero_grad()
            if inputs.size(0) <= 1:
                continue
            pred = mem_label[index]
            bsz = inputs.size(0)
            inputs = inputs.cuda()
            m_inputs = m_inputs.cuda()
            adjust_learning_rate(self.optimizer, (epoch - 1) * len(trainloader) + batchidx + 1, self.max_iters,
                                 warmup_iters=args.scheduler_warmup_epochs * len(trainloader))
            with autocast():
                outputs = self.netC(self.encoder(inputs))   
                classifier_loss = torch.tensor(0.).cuda()
                if args.par_su_cl > 0:
                    q = outputs.clone()
                    k = self.netC(self.encoder(m_inputs))
                    q = F.normalize(q, dim=-1)
                    k = F.normalize(k, dim=-1)

                    embeds_batch = torch.cat([q, k], dim=0)
                    pairwise_comp_batch = torch.matmul(embeds_batch, embeds_batch.t())  # 2N*2N
                    maskSup_batch, maskUnsup_batch = self.mask_estimation(selected_pairs, index, bsz)  # masksup_batch: 2N*2N
                    logits_mask_batch = (torch.ones_like(maskSup_batch) - torch.eye(2 * bsz).cuda())
                    ## Negatives mask, i.e. all except self-contrast sample
                    # sel_mask = (maskSup_batch[:bsz].sum(1)) < 1
                    # print(f'sum of sel_mask for low-confident samples:{sel_mask.sum()}')

                    loss_sup = args.par_su_cl * self.Supervised_ContrastiveLearning_loss(pairwise_comp_batch,
                                                                                        maskSup_batch, maskUnsup_batch,
                                                                                        logits_mask_batch, bsz)
                else:
                    loss_sup = 0

                classifier_loss += loss_sup
                if args.par_noisy_cls > 0:
                    if args.sel_cls:
                        assert trainSelloader is not None
                        try:
                            img, labels, _ = next(train_sel_iter)
                        except StopIteration:
                            train_sel_iter = iter(trainSelloader)
                            img, labels, _ = next(train_sel_iter)
                        img = img.cuda()
                        labels = labels.cuda()
                        sel_output = self.netC(self.encoder(img))
                        classifier_loss += nn.CrossEntropyLoss()(sel_output, labels) * args.par_noisy_cls
                    else:
                        cls_loss = nn.CrossEntropyLoss()(outputs, pred)
                        cls_loss *= args.par_noisy_cls
                        if epoch == 1 and args.dset == "VISDA-C":
                            cls_loss *= 0
                        classifier_loss += cls_loss

                if args.par_noisy_ent > 0:
                    softmax_out = nn.Softmax(dim=1)(outputs)  # (N,k)
                    entropy_loss = torch.mean(Loss.Entropy(softmax_out))
                    im_loss = entropy_loss * args.par_noisy_ent
                    msoftmax = softmax_out.mean(dim=0)  # 降维（K，）
                    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                    im_loss -= gentropy_loss * args.par_noisy_ent
                    classifier_loss += im_loss
            
            self.scaler.scale(classifier_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        return classifier_loss.item()


    def mask_estimation(self, selected_pairs, index, bsz):
        # 根据selected_Pairs进行mask的生成
        temp_graph = selected_pairs[index][:, index]
        # Create mask without diagonal to avoid augmented view, i.e. this is supervised mask
        maskSup_batch = temp_graph.float().cuda()
        maskSup_batch[torch.eye(bsz) == 1] = 0  # 去掉样本自身正例对
        maskSup_batch = maskSup_batch.repeat(2, 2)
        maskSup_batch[torch.eye(2 * bsz) == 1] = 0  # remove self-contrast case

        maskUnsup_batch = torch.eye(bsz, dtype=torch.float32).cuda()
        maskUnsup_batch = maskUnsup_batch.repeat(2, 2)
        maskUnsup_batch[torch.eye(2 * bsz) == 1] = 0  ##remove self-contrast (weak-to-weak, strong-to-strong) case #2B*2B
        return maskSup_batch, maskUnsup_batch


    def Supervised_ContrastiveLearning_loss(self, pairwise_comp, maskSup, maskUnsup, logits_mask, bsz):
        # maskUnsup是每个样本的一对正例
        # pairwise_comp:2N*2N,q,k合并
        logits = torch.div(pairwise_comp, args.su_cl_t)
        exp_logits = torch.exp(logits) * logits_mask

        if args.scheduler_warmup_epochs == 1:
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
            ## Approximation for numerical stability taken from supervised contrastive learning
        else:
            log_prob = torch.log(torch.exp(logits) + 1e-7) - torch.log(exp_logits.sum(1, keepdim=True) + 1e-7)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos_sup = (maskSup * log_prob).sum(1) / (maskSup.sum(1) + maskUnsup.sum(1))
        mean_log_prob_pos_unsup = (maskUnsup * log_prob).sum(1) / (maskSup.sum(1) + maskUnsup.sum(1))

        lossa = -mean_log_prob_pos_unsup[:int(len(mean_log_prob_pos_unsup) / 2)] \
                - mean_log_prob_pos_sup[:int(len(mean_log_prob_pos_sup) / 2)]
        lossb = -mean_log_prob_pos_unsup[int(len(mean_log_prob_pos_unsup) / 2):] \
                - mean_log_prob_pos_sup[int(len(mean_log_prob_pos_sup) / 2):]
        loss = torch.cat((lossa, lossb))
        loss = loss.view(2, bsz).mean(dim=0)

        loss = ((maskSup[:bsz].sum(1)) > 0) * (loss.view(bsz))  # 得是maskSup可以的
        return loss.mean()

    def start_train(self):
        param_group = []
        for k, v in self.encoder.netF.named_parameters():
            if args.lr_decay1 > 0:
                if v.requires_grad:
                    param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
            else:
                v.requires_grad = False
        for k, v in self.encoder.netB.named_parameters():
            if args.lr_decay2 > 0:
                if v.requires_grad:
                    param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
            else:
                v.requires_grad = False
        for k, v in self.netC.named_parameters():
            v.requires_grad = False

        optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=1e-3, nesterov=True)
        self.optimizer = op_copy(optimizer)
        acc_final = self.forward()
        return acc_final

    def forward(self):
        for epoch in range(1, args.max_epoch + 1):
            self.encoder.eval()
            self.netC.eval()
            mem_label, all_fea, initc, all_label, all_output = self.obtain_label(False)
            # all_fea: normalized, tensor type(grad cut)
            mem_label = torch.from_numpy(mem_label).cuda()
            self.encoder.train()
            self.netC.train()

            if epoch <= args.warmup_epochs:
                # only runned when dataset == visda-C
                classifier_loss = self.train_uns(epoch, cosine_warmup)

            elif epoch > args.warmup_epochs:
                selected_examples, selected_pairs = pair_selection_v1(args.k_val,
                                                                   self.loader['test'],
                                                                   mem_label, args.class_num,
                                                                   args.cos_t,
                                                                   args.knn_times,
                                                                   all_fea,
                                                                   balance_class=args.balance_class,
                                                                   sel_ratio=args.sel_ratio,
                                                                   epoch=epoch)

                # calculate pseudo-label accuracy of selected_examples
                self.encoder.eval()
                self.netC.eval()
                self.cal_sel_acc(real_labels=all_label, mem_labels=mem_label,
                                 selected_samples=selected_examples)

                # use the selected pseudo-labels to build a dataloader train_sel_loader to supervise training
                self.encoder.train()
                self.netC.train()
                txt_tar = open(args.t_dset_path).readlines()
                pseudo_dataset = Pseudo_dataset(txt_tar, mem_label.cpu().numpy(), transform=image_train())
                train_sel_loader = DataLoader(pseudo_dataset, batch_size=args.batch_size, num_workers=args.worker,
                                              pin_memory=True,
                                              sampler=torch.utils.data.WeightedRandomSampler(selected_examples,
                                                                                             len(selected_examples)))

                classifier_loss = self.train_su_cl(epoch, self.loader['two_train'], train_sel_loader,
                                                   mem_label, selected_pairs, cosine_warmup)

            # evaluate accuracy every epoch
            self.encoder.eval()
            self.netC.eval()
            if args.dset == 'VISDA-C':
                acc_s_te, acc_list = self.cal_acc(True)
                log_str = f'Task: {args.name}, epoch:{epoch}/{args.max_epoch}; Accuracy = {acc_s_te:.2f};' \
                          f'Loss = {classifier_loss:.2f}; \n {acc_list}'
            else:
                acc_s_te = self.cal_acc(False)
                log_str = f'Task: {args.name}, epoch:{epoch}/{args.max_epoch}; Accuracy = {acc_s_te:.2f} ;' \
                          f'Loss = {classifier_loss:.2f} '
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')


        if args.issave:
            args.save_dir = osp.join(args.output_dir, f'acc_{acc_s_te:.2f}_{args.savename}')
            if not osp.exists(args.save_dir):
                os.system('mkdir -p ' + args.save_dir)
            if not osp.exists(args.save_dir):
                os.mkdir(args.save_dir)
            torch.save(self.encoder.netF.state_dict(),
                       osp.join(args.save_dir, "target_F.pt"))
            torch.save(self.encoder.netB.state_dict(),
                       osp.join(args.save_dir, "target_B.pt"))
            torch.save(self.netC.state_dict(),
                       osp.join(args.save_dir, "target_C.pt"))
        return round(acc_s_te,1)
        
        
    def cal_sel_acc(self, real_labels, mem_labels, selected_samples):
        # accuracy of selected samples
        with torch.no_grad():
            idx_selected = selected_samples.nonzero().squeeze()
            sel_mem_labels = mem_labels[idx_selected]
            sel_real_labels = real_labels[idx_selected]
            sel_acc = (sel_real_labels == sel_mem_labels).sum().item() / selected_samples.sum().item()
        logstr = f'selection samples accuracy:{100 * sel_acc:.2f}%'
        print(logstr)
        args.out_file.write(logstr+'\n')
        args.out_file.flush()

    def cal_acc(self, flag=False):
        start_test = True
        with torch.no_grad():
            iter_train = iter(self.loader['test'])
            for i in range(len(self.loader['test'])):
                data = iter_train.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                outputs = self.netC(self.encoder(inputs))
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
            aacc = acc.mean()
            aa = [str(np.round(i, 2)) for i in acc]
            acc = ' '.join(aa)
            return aacc, acc  # acc是单类的正确率 aacc是acc的均值
        else:
            return accuracy * 100

    def obtain_label(self, return_dist=False):
        start_test = True
        with torch.no_grad():
            iter_train = iter(self.loader['test'])
            for _ in range(len(self.loader['test'])):
                data = iter_train.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                feas = self.encoder(inputs)
                outputs = self.netC(feas)
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
    parser.add_argument('--output_src', type=str, default='res/ckps/source')
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
        args.run_all = False
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    folder = '../DATASETS/'
    
    def tem_run():
        startTime = datetime.datetime.now()
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
        args.name = names[args.s][0].upper() + names[args.t][0].upper()
        
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
        
    args.savename = f'{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}_par_n_ent_{args.par_noisy_ent}_par_su_cl_{args.par_su_cl}_tau2_{args.su_cl_t}_kval_{args.k_val}_selr_{args.sel_ratio}_knnt_{args.knn_times}'
    args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    print(f'args:{args}')
    args.out_file.flush()
    
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
        

    
    