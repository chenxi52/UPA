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
        self.args = args
        self.ttransforms = image_train()

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

                softmax_out = nn.Softmax(dim=1)(outputs) 
                entropy_loss = torch.mean(Loss.Entropy(softmax_out))
                im_loss = entropy_loss * self.args.par_ent
                msoftmax = softmax_out.mean(dim=0)  
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + self.args.epsilon))
                im_loss -= gentropy_loss * self.args.par_ent
                classifier_loss += im_loss

            self.scaler.scale(classifier_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
        return classifier_loss.item()


    def train_su_cl(self, epoch, trainloader, trainSelloader, mem_label, selected_pairs, adjust_learning_rate):
        if trainSelloader:
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
                    pairwise_comp_batch = torch.matmul(embeds_batch, embeds_batch.t()) 
                    maskSup_batch, maskUnsup_batch = self.mask_estimation(selected_pairs, index, bsz) 
                    logits_mask_batch = (torch.ones_like(maskSup_batch) - torch.eye(2 * bsz).cuda())

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
                    softmax_out = nn.Softmax(dim=1)(outputs)  
                    entropy_loss = torch.mean(Loss.Entropy(softmax_out))
                    im_loss = entropy_loss * self.args.par_noisy_ent
                    msoftmax = softmax_out.mean(dim=0)  
                    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + self.args.epsilon))
                    im_loss -= gentropy_loss * self.args.par_noisy_ent
                    classifier_loss += im_loss
            
            self.scaler.scale(classifier_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        return classifier_loss.item()


    def mask_estimation(self, selected_pairs, index, bsz):
        temp_graph = selected_pairs[index][:, index]
        # Create mask without diagonal to avoid augmented view, i.e. this is supervised mask
        maskSup_batch = temp_graph.float().cuda()
        maskSup_batch[torch.eye(bsz) == 1] = 0  
        maskSup_batch = maskSup_batch.repeat(2, 2)
        maskSup_batch[torch.eye(2 * bsz) == 1] = 0  # remove self-contrast case

        maskUnsup_batch = torch.eye(bsz, dtype=torch.float32).cuda()
        maskUnsup_batch = maskUnsup_batch.repeat(2, 2)
        maskUnsup_batch[torch.eye(2 * bsz) == 1] = 0  #remove self-contrast (weak-to-weak, strong-to-strong) case #2B*2B
        return maskSup_batch, maskUnsup_batch


    def Supervised_ContrastiveLearning_loss(self, pairwise_comp, maskSup, maskUnsup, logits_mask, bsz):
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

    def start_train(self):
        param_group = []
        for k, v in self.encoder.netF.named_parameters():
            if self.args.lr_decay1 > 0:
                if v.requires_grad:
                    param_group += [{'params': v, 'lr': self.args.lr * self.args.lr_decay1}]
            else:
                v.requires_grad = False
        for k, v in self.encoder.netB.named_parameters():
            if self.args.lr_decay2 > 0:
                if v.requires_grad:
                    param_group += [{'params': v, 'lr': self.args.lr * self.args.lr_decay2}]
            else:
                v.requires_grad = False
        for k, v in self.netC.named_parameters():
            v.requires_grad = False

        optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=1e-3, nesterov=True)
        self.optimizer = op_copy(optimizer)
        acc_final = self.forward()
        return acc_final

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
                                                                   self.loader['test'],
                                                                   mem_label, self.args.class_num,
                                                                   self.args.cos_t,
                                                                   self.args.knn_times,
                                                                   all_fea,
                                                                   balance_class=self.args.balance_class,
                                                                   sel_ratio=self.args.sel_ratio,
                                                                   )

                # calculate pseudo-label accuracy of selected_examples
                self.encoder.eval()
                self.netC.eval()
                self.cal_sel_acc(real_labels=all_label, mem_labels=mem_label,
                                 selected_samples=selected_examples)

                # use the selected pseudo-labels to build a dataloader train_sel_loader to supervise training
                self.encoder.train()
                self.netC.train()
                txt_tar = open(self.args.t_dset_path).readlines()
                pseudo_dataset = Pseudo_dataset(txt_tar, mem_label.cpu().numpy(), transform=self.ttransforms, append_root=self.args.append_root)
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
            torch.save(self.encoder.netF.state_dict(),
                       osp.join(self.args.save_dir, "target_F.pt"))
            torch.save(self.encoder.netB.state_dict(),
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
            return aacc, acc  
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
        if self.args.distance == 'cosine':
            all_fea = torch.cat([all_fea, torch.ones(all_fea.size(0), 1)], 1) 
            all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t() 
            all_fea = all_fea.float().cpu().numpy()

        K = all_output.size(1)
        aff = all_output.float().cpu().numpy() 
        initc = aff.transpose().dot(all_fea)  
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])  
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count > 0)
        labelset = labelset[0]

        dd = cdist(all_fea, initc[labelset], self.args.distance) 
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

        for round in range(1):
            aff = np.eye(K)[pred_label] 
            initc = aff.transpose().dot(all_fea) 
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_fea, initc[labelset], self.args.distance)
            pred_label = dd.argmin(axis=1)
            pred_label = labelset[pred_label] 

        min_dist = dd.min(axis=1)  
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
        log_str = f'Task:{self.args.name}   Accuracy = {accuracy * 100:.2f}% -> {acc * 100:.2f}%'

        self.args.out_file.write(log_str + '\n')
        self.args.out_file.flush()
        print(log_str + '\n')

        if return_dist:
            return pred_label.astype('int'), min_dist
        else:
            return pred_label.astype('int'), torch.from_numpy(all_fea[:, :-1]).cuda(), initc[:, :-1], all_label.float().cuda(), all_output.cuda()

