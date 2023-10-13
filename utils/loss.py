import numpy
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def Normalized_entropy(input):
    epsilon=1e-5
    entropy=-input*torch.log(input+epsilon)
    entropy=entropy/torch.log(input.size(-1))
    entropy=torch.sum(entropy,dim=1)
    return entropy

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))       
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target) 

def DANN(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss

def crossEntropySmooth(inputs,target):
    inputs=nn.LogSoftmax(dim=1)(inputs)
    target=nn.Softmax(dim=1)(target).long()
    loss=(-target*inputs).sum(dim=-1).mean()
    return loss

def lossc(alphas,alphat,sourcep,targetp):
    softmaxp = F.softmax(sourcep,dim=1)#(64,31)
    softmaxt = F.softmax(targetp,dim=1)#(64,31)
    loss = torch.sum(-softmaxp*torch.log(softmaxp)*alphas-softmaxt*torch.log(softmaxt)*alphat,dim=1)#(64)
    return loss.mean()

def losscplus(alphas,alphat,pst_s,pst_t):
    part1=alphas*torch.log(torch.sum(pst_s,dim=1,keepdim=True))#(64,1)*(64,1)->(64,1)
    part2=alphat*torch.log(torch.sum(pst_t,dim=1,keepdim=True))#(64,1)*(64,1)->(64,1)
    losss=part1+part2
    loss=-(torch.sum(losss,dim=1))#(64)
    return loss.mean()

def losscplusplus(alphas,alphat,pst_s,pst_t):
    part1 = alphat * torch.log(torch.sum(pst_s, dim=1,keepdim=True))#(64,1)*(64,1)->(64,1)
    part2 = alphas * torch.log(torch.sum(pst_t, dim=1,keepdim=True))#(64,1)*(64,1)->(64,1)
    loss = -(torch.sum(part1 + part2,dim=1))#(64)
    return loss.mean()

def lossr(logits,target):
    CEloss=nn.CrossEntropyLoss()
    return CEloss(logits,target)

def lossp(miut,lt,sourcep):
    softmaxp=F.softmax(sourcep,dim=1)#(64,31)
    batch_size=sourcep.shape[0]
    loss=torch.tensor(0.).to('cuda:0')
    ss=torch.mm(softmaxp,softmaxp.t())#(64,64)
    if miut<=lt:
        return softmaxp.sum(dim=0).sum(dim=0)*(torch.tensor(0).to('cuda:0'))
    for i in range(batch_size):
        for j in range(batch_size):
            if j==i:
                continue
            sij=ss[i][j]
            siv=ss[i]#(64)
            index=torch.tensor(numpy.array(list(map(int,[i!=v for v in range(batch_size)])))).to('cuda:0')#(64)
            target=torch.tensor(numpy.array(list(map(int,[(siv[i]>miut or siv[i]<lt) for i in range(batch_size)])))).to('cuda:0')#(64)
            eps=-torch.log(torch.exp(sij)/(torch.exp(siv)*index*target).sum(dim=0))#(64)
            loss+=(sij>miut)*eps
    return loss/batch_size

import torch
import torch.nn.functional as F

def info_nce_loss(args, features_q, features_k):
    labels = torch.arange(features_q.size(0))
    labels = labels.cuda()

    features_q = F.normalize(features_q, dim=1)
    features_k = F.normalize(features_k, dim=1)

    similarity_matrix = torch.matmul(features_q, features_k.T)  # B*256,256*B=BB
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()  # BB
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[mask.bool()].view(labels.shape[0], -1)  # B,1

    # select only the negatives the negatives
    negatives = similarity_matrix[~mask.bool()].view(labels.shape[0], -1)  # B,B-1

    logits = torch.cat([positives, negatives], dim=1)  # BB
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()  # B

    logits = logits / args.temperature
    return logits, labels

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07,distributed=False):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.distributed=distributed
    def forward(self, features, labels=None, mask=None,weight=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device('cuda') if not self.distributed else features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        # print(f'mask:{mask}')
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)#2N*C
        # print(f'contrast feature:{contrast_feature}')
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        # print(f'anchor_feature shape:{anchor_feature.shape}')#[6,5]
        # compute logits,2N*2N
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()
        logits=anchor_dot_contrast
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # print(f'repeated mask:{mask}')
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        # print(f'logits_mask:{logits_mask}')
        mask = mask * logits_mask
        # print(f'mask:{mask}')
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))#2N*2N
        if weight is not None:
            weight = weight.repeat(anchor_count)  # 2N*1
            log_prob=weight*log_prob#2N*1 * 2N*2N

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

def kl_div_with_logit(q_logit, p_logit):

    q = F.softmax(q_logit, dim=1)
    logq = F.log_softmax(q_logit, dim=1)
    logp = F.log_softmax(p_logit, dim=1)

    qlogq = ( q *logq).sum(dim=1).mean(dim=0)
    qlogp = ( q *logp).sum(dim=1).mean(dim=0)

    return qlogq - qlogp


def _l2_normalize(d):

    d = d.numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2, 3))).reshape((-1, 1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)


def vat_loss(args,netF,netB,netC, ul_x, ul_y, xi=1e-6, eps=2.5, num_iters=1):
    # find r_adv
    # print(f'ul_x size:{ul_x.size(0)}')
    if ul_x.size(0)<=2*len(args.gpu_id.split(',')):
        print('batch for each gpu <= 1')
        return 0,0
    d = torch.Tensor(ul_x.size()).normal_()
    for i in range(num_iters):
        netF.zero_grad()
        netB.zero_grad()
        netC.zero_grad()
        d = xi *_l2_normalize(d)
        d = Variable(d.cuda(), requires_grad=True)
        y_hat = netC(netB(netF(ul_x + d)))
        delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)#
        delta_kl.backward()
        d = d.grad.data.clone().cpu()

    d = _l2_normalize(d)
    d = Variable(d.cuda())
    r_adv = eps *d
    # compute lds
    x_adv=ul_x + r_adv.detach()
    y_hat = netC(netB(netF(x_adv)))
    delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)

    # f_out_ulx=model[1](model[0](ul_x))
    # print(f'f_out_ulx:{f_out_ulx.size()}')
    # f_out_adv=model[1](model[0](x_adv))
    # f_norm=torch.mean(torch.norm(torch.abs(f_out_ulx.detach()-f_out_adv),p=2,dim=1))
    # f_norm=0

    return delta_kl,y_hat

def entropy_loss(ul_y):
    p = F.softmax(ul_y, dim=1)
    return -(p*F.log_softmax(ul_y, dim=1)).sum(dim=1).mean(dim=0)
# from model.gather import GatherLayer
# class NT_Xent(nn.Module):
#     def __init__(self, batch_size, temperature, world_size):
#         super(NT_Xent, self).__init__()
#         self.batch_size = batch_size
#         self.temperature = temperature
#         self.world_size = world_size
#         self.mask=self.mask_correlated_samples(self.batch_size,self.world_size)
#         self.criterion = nn.CrossEntropyLoss(reduction="sum")
#         self.similarity_f = nn.CosineSimilarity(dim=2)
#
#     def mask_correlated_samples(self, batch_size, world_size):
#         N = 2 * batch_size * world_size
#         mask = torch.ones((N, N), dtype=bool)
#         mask = mask.fill_diagonal_(0)
#         for i in range(batch_size * world_size):
#             mask[i, batch_size * world_size + i] = 0
#             mask[batch_size * world_size + i, i] = 0
#         print(f'mask:{mask.shape}')#24,24
#         return mask
#
#     def forward(self, z_i, z_j):
#         """
#         We do not sample negative examples explicitly.
#         Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
#         """
#         N = 2 * self.batch_size * self.world_size
#         print(f'N:{N}')#24
#         if self.world_size > 1:
#             z_i = torch.cat(GatherLayer.apply(z_i), dim=0)
#             z_j = torch.cat(GatherLayer.apply(z_j), dim=0)
#         z = torch.cat((z_i, z_j), dim=0)
#         print(f'gather z shape:{z.shape}')#24,1
#         print(f'gather z:{z}')
#         sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
#         print(f'sim:{sim}')
#         print(f'sim.shape:{sim.size()}')#24,24
#         sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
#         print(f'sim_i_j:{sim_i_j}')
#         sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)
#         print(f'sim_i_j.shape:{sim_i_j.shape}')#12
#         # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
#         # self.mask = self.mask_correlated_samples(i_batch_size,j_batch_size, world_size)
#         positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
#         negative_samples = sim[self.mask].reshape(N, -1)
#
#         labels = torch.zeros(N).to(positive_samples.device).long()
#         logits = torch.cat((positive_samples, negative_samples), dim=1)
#         loss = self.criterion(logits, labels)
#         loss /= N
#         return loss
def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, max_epochs=30, lambda_u=75):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, lambda_u * linear_rampup(epoch, max_epochs)