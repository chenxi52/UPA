import os
import os.path as osp
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import torch
import numpy as np
from model import network
import torch.nn.functional as F
from loaders.target_loader import data_load
from loaders.data_list import Pseudo_dataset
from torch.utils.data import DataLoader
from utils.adjust_par import op_copy, cosine_warmup
from utils.tools import print_args, image_train
from utils import loss as Loss
from utils.utils_noise import pair_selection_v1
from torchvision import transforms
from loaders.data_load import mnist, svhn, usps
from utils.tools import GaussianBlur

from torch.cuda.amp import autocast

def mocov2(resize_size=256,crop_size=224,alexnet=False,channel=3):
    if channel==3:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
        augmentation = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
       
    elif channel==1:
        normalize = transforms.Normalize(mean=[0.5],std= [0.5])
        augmentation = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.)),
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

def digit_load(args, return_ttransform=False): 
    train_bs = args.batch_size
    if args.dset == 's2m':
        train_source = svhn.SVHN('./data/svhn/', split='train', download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))
        test_source = svhn.SVHN('./data/svhn/', split='test', download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))  
        train_target = mnist.MNIST_idx('./data/mnist/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))      
        test_target = mnist.MNIST('./data/mnist/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))
        tt_transform =transforms.Compose([
                    transforms.Resize(32),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        two_target = mnist.MNIST_idx('./data/mnist/', train=True, download=True,
                transform=tt_transform,
                transform2=mocov2(resize_size=32, crop_size=32, channel=3))
    elif args.dset == 'u2m':
        train_source = usps.USPS('./data/usps/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(28, padding=4),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
        test_source = usps.USPS('./data/usps/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(28, padding=4),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))    
        train_target = mnist.MNIST_idx('./data/mnist/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))    
        test_target = mnist.MNIST('./data/mnist/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
        tt_transform =transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
        two_target = mnist.MNIST_idx('./data/mnist/', train=True, download=True,
                transform=tt_transform,
                transform2=mocov2(resize_size=16, crop_size=28, channel=1))
    elif args.dset == 'm2u': # mnist: 28x28
        train_source = mnist.MNIST('./data/mnist/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
        test_source = mnist.MNIST('./data/mnist/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))

        train_target = usps.USPS_idx('./data/usps/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
        test_target = usps.USPS('./data/usps/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
        tt_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
        two_target = usps.USPS_idx('./data/usps/', train=True, download=True,
                transform=tt_transform,
                transform2=mocov2(resize_size=28, crop_size=28, channel=1))
    dset_loaders = {}
    dset_loaders["source_tr"] = DataLoader(train_source, batch_size=train_bs, shuffle=True, 
        num_workers=args.worker, drop_last=False)
    dset_loaders["source_te"] = DataLoader(test_source, batch_size=train_bs*2, shuffle=True, 
        num_workers=args.worker, drop_last=False)
    dset_loaders["target"] = DataLoader(train_target, batch_size=train_bs, shuffle=True, 
        num_workers=args.worker, drop_last=False)
    dset_loaders["target_te"] = DataLoader(train_target, batch_size=train_bs, shuffle=False, 
        num_workers=args.worker, drop_last=False)
    dset_loaders["test"] = DataLoader(test_target, batch_size=train_bs*2, shuffle=False, 
        num_workers=args.worker, drop_last=False)
    dset_loaders["two_train"] = DataLoader(two_target, batch_size=train_bs, shuffle=True,
        num_workers=args.worker, drop_last=False)
    if not return_ttransform:
        return dset_loaders
    else:
        return dset_loaders, tt_transform

class dig_Upa(object):
    def __init__(self, args, netB, netC, netF):
        super().__init__()
        self.args = args
        self.netF = netF
        self.netB = netB
        self.encoder = nn.Sequential(self.netF, self.netB).cuda()
        self.netC = netC.cuda()
        self.loader, self.ttransforms = digit_load(args, return_ttransform=True)
        self.max_iters = args.max_epoch * len(self.loader["target"])
        self.scaler = torch.cuda.amp.GradScaler()
        param_group = []
        for k, v in self.netF.named_parameters():
            param_group += [{'params': v, 'lr': args.lr}]
        for k, v in self.netB.named_parameters():
            param_group += [{'params': v, 'lr': args.lr}]
        for k, v in self.netC.named_parameters():
            v.requires_grad = False
        self.optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=1e-3, nesterov=True)
        self.optimizer = op_copy(self.optimizer)

    def start_train(self):
        acc_final = self.forward()
        return acc_final


    def train_uns(self, epoch, adjust_learning_rate):
        for batchidx, (inputs, _, _, tar_idx) in enumerate(self.loader['two_train']):
            self.optimizer.zero_grad()
            if inputs.size(0) < 2:
                continue
            inputs = inputs.cuda()

            adjust_learning_rate(self.optimizer, (epoch - 1) * len(self.loader['two_train']) + batchidx + 1,
                                 self.max_iters,
                                 warmup_iters=self.args.scheduler_warmup_epochs * len(self.loader['two_train']))
            with autocast():
                features = self.encoder(inputs)
                outputs = self.netC(features)
                classifier_loss = torch.tensor(0.).cuda()

                softmax_out = nn.Softmax(dim=1)(outputs)  # (N,k)
                entropy_loss = torch.mean(Loss.Entropy(softmax_out))
                im_loss = entropy_loss * self.args.par_ent
                msoftmax = softmax_out.mean(dim=0)  # 降维（K，）
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + self.args.epsilon))
                im_loss -= gentropy_loss * self.args.par_ent
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
                                 warmup_iters=self.args.scheduler_warmup_epochs * len(trainloader))
            with autocast():
                outputs = self.netC(self.encoder(inputs))   
                classifier_loss = torch.tensor(0.).cuda()
                if self.args.par_su_cl > 0:
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

                    loss_sup = self.args.par_su_cl * self.Supervised_ContrastiveLearning_loss(pairwise_comp_batch,
                                                                                        maskSup_batch, maskUnsup_batch,
                                                                                        logits_mask_batch, bsz)
                else:
                    loss_sup = 0

                classifier_loss += loss_sup
                if self.args.par_noisy_cls > 0:
                    if self.args.sel_cls:
                        assert trainSelloader is not None
                        try:
                            img, labels, _ = next(train_sel_iter)
                        except StopIteration:
                            train_sel_iter = iter(trainSelloader)
                            img, labels, _ = next(train_sel_iter)
                        img = img.cuda()
                        labels = labels.cuda()
                        sel_output = self.netC(self.encoder(img))
                        classifier_loss += nn.CrossEntropyLoss()(sel_output, labels) * self.args.par_noisy_cls
                    else:
                        cls_loss = nn.CrossEntropyLoss()(outputs, pred)
                        cls_loss *= self.args.par_noisy_cls
                        if epoch == 1 and self.args.dset == "VISDA-C":
                            cls_loss *= 0
                        classifier_loss += cls_loss

                if self.args.par_noisy_ent > 0:
                    softmax_out = nn.Softmax(dim=1)(outputs)  # (N,k)
                    entropy_loss = torch.mean(Loss.Entropy(softmax_out))
                    im_loss = entropy_loss * self.args.par_noisy_ent
                    msoftmax = softmax_out.mean(dim=0)  # 降维（K，）
                    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + self.args.epsilon))
                    im_loss -= gentropy_loss * self.args.par_noisy_ent
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
        logits = torch.div(pairwise_comp, self.args.su_cl_t)
        exp_logits = torch.exp(logits) * logits_mask

        if self.args.scheduler_warmup_epochs == 1:
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


    def forward(self):
        for epoch in range(1, self.args.max_epoch + 1):
            self.encoder.eval()
            self.netC.eval()
            mem_label, all_fea, initc, all_label, all_output = self.obtain_label(False)
            # all_fea: normalized, tensor type(grad cut)
            mem_label = torch.from_numpy(mem_label).cuda()
            self.encoder.train()
            self.netC.train()

            if epoch <= self.args.warmup_epochs:
                # only runned when dataset == visda-C
                classifier_loss = self.train_uns(epoch, cosine_warmup)

            elif epoch > self.args.warmup_epochs:
                selected_examples, selected_pairs = pair_selection_v1(self.args.k_val,
                                                                   self.loader['target_te'],
                                                                   mem_label, self.args.class_num,
                                                                   self.args.cos_t,
                                                                   self.args.knn_times,
                                                                   all_fea,
                                                                   balance_class=self.args.balance_class,
                                                                   sel_ratio=self.args.sel_ratio,
                                                                   epoch=epoch)

                # calculate pseudo-label accuracy of selected_examples
                self.encoder.eval()
                self.netC.eval()
                self.cal_sel_acc(real_labels=all_label, mem_labels=mem_label,
                                 selected_samples=selected_examples)

                # use the selected pseudo-labels to build a dataloader train_sel_loader to supervise training
                self.encoder.train()
                self.netC.train()
                # txt_tar = open(self.args.t_dset_path).readlines()
                if self.args.dset == 's2m' or self.args.dset== 'u2m':
                    pseudo_dataset = mnist.MNIST_idx('./data/mnist/', train=True, download=True,
                                    transform=self.ttransforms, mem_label=mem_label.cpu())
                elif self.args.dset == 'm2u':
                    pseudo_dataset = usps.USPS_idx('./data/usps/', train=True, download=True,
                                    transform=self.ttransforms, mem_label=mem_label.cpu())
                # pseudo_dataset = Pseudo_dataset(txt_tar, mem_label.cpu().numpy(), transform=self.ttransforms)
                
                train_sel_loader = DataLoader(pseudo_dataset, batch_size=self.args.batch_size, num_workers=self.args.worker,
                                              pin_memory=True,
                                              sampler=torch.utils.data.WeightedRandomSampler(selected_examples,
                                                                                             len(selected_examples)))

                classifier_loss = self.train_su_cl(epoch, self.loader['two_train'], train_sel_loader,
                                                   mem_label, selected_pairs, cosine_warmup)

            # evaluate accuracy every epoch
            self.encoder.eval()
            self.netC.eval()
            if self.args.dset == 'VISDA-C':
                acc_s_te, acc_list = self.cal_acc(True)
                log_str = f'Task: {self.args.name}, epoch:{epoch}/{self.args.max_epoch}; Accuracy = {acc_s_te:.2f};' \
                          f'Loss = {classifier_loss:.2f}; \n {acc_list}'
            else:
                acc_s_te = self.cal_acc(False)
                log_str = f'Task: {self.args.name}, epoch:{epoch}/{self.args.max_epoch}; Accuracy = {acc_s_te:.2f} ;' \
                          f'Loss = {classifier_loss:.2f} '
            self.args.out_file.write(log_str + '\n')
            self.args.out_file.flush()
            print(log_str + '\n')


        if self.args.issave:
            self.args.save_dir = osp.join(self.args.output_dir, f'acc_{acc_s_te:.2f}_{self.args.savename}')
            if not osp.exists(self.args.save_dir):
                os.system('mkdir -p ' + self.args.save_dir)
            if not osp.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            torch.save(self.netF.state_dict(),
                       osp.join(self.args.save_dir, "target_F.pt"))
            torch.save(self.netB.state_dict(),
                       osp.join(self.args.save_dir, "target_B.pt"))
            torch.save(self.netC.state_dict(),
                       osp.join(self.args.save_dir, "target_C.pt"))
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
        self.args.out_file.write(logstr+'\n')
        self.args.out_file.flush()

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
            iter_train = iter(self.loader['target_te'])
            for _ in range(len(self.loader['target_te'])):
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
        if self.args.distance == 'cosine':
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

        dd = cdist(all_fea, initc[labelset], self.args.distance)  # cdist(N*f+1,k*f+1)-->N*k+1
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

        for round in range(1):
            aff = np.eye(K)[pred_label]  # N*K
            initc = aff.transpose().dot(all_fea)  # K*(f+1)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_fea, initc[labelset], self.args.distance)
            pred_label = dd.argmin(axis=1)
            pred_label = labelset[pred_label]  # (N,)

        min_dist = dd.min(axis=1)  # 样本距离自己的聚类中心的距离
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
        log_str = f'Task:{self.args.name}   Accuracy = {accuracy * 100:.2f}% -> {acc * 100:.2f}%'
        # accuracy:目前的模型预测成功率；
        # acc：pseudo label和true label相似度

        self.args.out_file.write(log_str + '\n')
        self.args.out_file.flush()
        print(log_str + '\n')

        if return_dist:
            return pred_label.astype('int'), min_dist
        else:
            return pred_label.astype('int'), torch.from_numpy(all_fea[:, :-1]).cuda(), initc[:, :-1], all_label.float().cuda(), all_output.cuda()

