from __future__ import print_function
import torch
import warnings
from scipy.spatial.distance import cdist
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from utils import loss as Loss
warnings.filterwarnings('ignore')


################################# MOIT #############################################
## Masks creation
## Unsupervised mask for batch and memory (note that memory also contains the current mini-batch)

def unsupervised_masks_estimation(mix_index1, mix_index2, bsz):
    labelsUnsup = torch.arange(bsz).long().unsqueeze(1).cuda()  # If no labels used, label is the index in mini-batch
    maskUnsup_batch = torch.eye(bsz, dtype=torch.float32).cuda()
    maskUnsup_batch = maskUnsup_batch.repeat(2, 2)
    maskUnsup_batch[torch.eye(2 * bsz) == 1] = 0  ##remove self-contrast case

    maskUnsup_mem = []

    ######################### Mixup additional mask: unsupervised term ######################
    # 记录minor image index 和major image index相同的index,就是没有标签下看image是否是同一个
    ## With no labels (labelUnsup is just the index in the mini-batch, i.e. different for each sample)
    quad1_unsup = torch.eq(labelsUnsup[mix_index1],
                           labelsUnsup.t()).float()  ##Minor label in 1st mini-batch part equal to mayor label in the first mini-batch part (note that mayor label of 1st and 2nd is the same as we force the original image to always be the dominant)
    quad2_unsup = torch.eq(labelsUnsup[mix_index1],
                           labelsUnsup.t()).float()  ##Minor label in 1st mini-batch part equal to mayor label in the second mini-batch part
    quad3_unsup = torch.eq(labelsUnsup[mix_index2],
                           labelsUnsup.t()).float()  ##Minor label in 2nd mini-batch part equal to mayor label in the first mini-batch part
    quad4_unsup = torch.eq(labelsUnsup[mix_index2],
                           labelsUnsup.t()).float()  ##Minor label in 2nd mini-batch part equal to mayor label in the second mini-batch part

    mask2_a_unsup = torch.cat((quad1_unsup, quad2_unsup), dim=1)
    mask2_b_unsup = torch.cat((quad3_unsup, quad4_unsup), dim=1)
    mask2Unsup_batch = torch.cat((mask2_a_unsup, mask2_b_unsup), dim=0)

    ## Make sure diagonal is zero (i.e. not taking as positive my own sample)
    mask2Unsup_batch[torch.eye(2 * bsz) == 1] = 0

    mask2Unsup_mem = []

    return maskUnsup_batch, maskUnsup_mem, mask2Unsup_batch, mask2Unsup_mem


def supervised_masks_estimation(labels, mix_index1, mix_index2, bsz):
    # 这里应该和经典的super-cl一致，直接根据label进行对比
    ###################### Supervised mask excluding augmented view ###############################
    labels = labels.contiguous().view(-1, 1)

    if labels.shape[0] != bsz:
        raise ValueError('Num of labels does not match num of features')

    ##Create mask without diagonal to avoid augmented view, i.e. this is supervised mask
    maskSup_batch = torch.eq(labels, labels.t()).float() - torch.eye(bsz, dtype=torch.float32).cuda()
    maskSup_batch = maskSup_batch.repeat(2, 2)
    maskSup_batch[torch.eye(2 * bsz) == 1] = 0  ##remove self-contrast case

    maskSup_mem = []

    ######################### Mixup additional mask: supervised term ######################
    ## With labels
    quad1_sup = torch.eq(labels[mix_index1],
                         labels.t()).float()  ##Minor label in 1st mini-batch part equal to mayor label in the first mini-batch part (note that mayor label of 1st and 2nd is the same as we force the original image to always be the mayor/dominant)
    quad2_sup = torch.eq(labels[mix_index1],
                         labels.t()).float()  ##Minor label in 1st mini-batch part equal to mayor label in the second mini-batch part
    quad3_sup = torch.eq(labels[mix_index2],
                         labels.t()).float()  ##Minor label in 2nd mini-batch part equal to mayor label in the first mini-batch part
    quad4_sup = torch.eq(labels[mix_index2],
                         labels.t()).float()  ##Minor label in 2nd mini-batch part equal to mayor label in the second mini-batch part

    mask2_a_sup = torch.cat((quad1_sup, quad2_sup), dim=1)
    mask2_b_sup = torch.cat((quad3_sup, quad4_sup), dim=1)
    mask2Sup_batch = torch.cat((mask2_a_sup, mask2_b_sup), dim=0)

    ## Make sure diagonal is zero (i.e. not taking as positive my own sample)
    mask2Sup_batch[torch.eye(2 * bsz) == 1] = 0

    mask2Sup_mem = []
    return maskSup_batch, maskSup_mem, mask2Sup_batch, mask2Sup_mem


def InterpolatedContrastiveLearning_loss(icl_t, aprox, pairwise_comp, maskSup, mask2Sup, maskUnsup, mask2Unsup,
                                         logits_mask, lam1, lam2, bsz):
    logits = torch.div(pairwise_comp, icl_t)
    exp_logits = torch.exp(logits) * logits_mask  # remove diagonal

    if aprox == 1:
        log_prob = logits - torch.log(exp_logits.sum(1,
                                                     keepdim=True))  ## Approximation for numerical stability taken from supervised contrastive learning
    else:
        log_prob = torch.log(torch.exp(logits) + 1e-10) - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)
    exp_logits2 = torch.exp(logits) * logits_mask  # remove diagonal

    if aprox == 1:
        log_prob2 = logits - torch.log(exp_logits2.sum(1,
                                                       keepdim=True))  ## Approximation for numerical stability taken from supervised contrastive learning
    else:
        log_prob2 = torch.log(torch.exp(logits) + 1e-10) - torch.log(exp_logits2.sum(1, keepdim=True) + 1e-10)
    # compute mean of log-likelihood over positive (weight individual loss terms with mixing coefficients)

    mean_log_prob_pos_sup = (maskSup * log_prob).sum(1) / (maskSup.sum(1) + maskUnsup.sum(1))
    mean_log_prob_pos_unsup = (maskUnsup * log_prob).sum(1) / (maskSup.sum(1) + maskUnsup.sum(1))
    ## Second mixup term log-probs
    mean_log_prob_pos2_sup = (mask2Sup * log_prob2).sum(1) / (mask2Sup.sum(1) + mask2Unsup.sum(1))
    mean_log_prob_pos2_unsup = (mask2Unsup * log_prob2).sum(1) / (mask2Sup.sum(1) + mask2Unsup.sum(1))

    ## Weight first and second mixup term (both data views) with the corresponding mixing weight

    ##First mixup term. First mini-batch part. Unsupervised + supervised loss separated
    loss1a = -lam1 * mean_log_prob_pos_unsup[:int(len(mean_log_prob_pos_unsup) / 2)] - lam1 * mean_log_prob_pos_sup[
                                                                                              :int(
                                                                                                  len(mean_log_prob_pos_sup) / 2)]
    ##First mixup term. Second mini-batch part. Unsupervised + supervised loss separated
    loss1b = -lam2 * mean_log_prob_pos_unsup[int(len(mean_log_prob_pos_unsup) / 2):] - lam2 * mean_log_prob_pos_sup[
                                                                                              int(len(
                                                                                                  mean_log_prob_pos_sup) / 2):]
    ## All losses for first mixup term
    loss1 = torch.cat((loss1a, loss1b))

    ##Second mixup term. First mini-batch part. Unsupervised + supervised loss separated
    loss2a = -(1.0 - lam1) * mean_log_prob_pos2_unsup[:int(len(mean_log_prob_pos2_unsup) / 2)] - (
            1.0 - lam1) * mean_log_prob_pos2_sup[:int(len(mean_log_prob_pos2_sup) / 2)]
    ##Second mixup term. Second mini-batch part. Unsupervised + supervised loss separated
    loss2b = -(1.0 - lam2) * mean_log_prob_pos2_unsup[int(len(mean_log_prob_pos2_unsup) / 2):] - (
            1.0 - lam2) * mean_log_prob_pos2_sup[int(len(mean_log_prob_pos2_sup) / 2):]
    ## All losses secondfor first mixup term
    loss2 = torch.cat((loss2a, loss2b))
    ## Final loss (summation of mixup terms after weighting)
    loss = loss1 + loss2

    loss = loss.view(2, bsz).mean()
    return loss


def pair_selection(k_val, testloader, labels, class_num, cos_t, alpha, trainFeatures, balance_class=False,
                   sel_ratio=0, sel_from_all=True, two_knn=True, plot=False, epoch=-1, corrected=False, three_knn=False):
    '''
    kval: neighbors number of knn
    label: the mem_label, pseudo-labels obtained by feature prototypes
    '''

    similiar_graph_all = torch.zeros(len(testloader.dataset), len(testloader.dataset))

    trainNoisyLabels = labels.clone().cuda()
    train_new_labels = labels.clone().cuda()
    discrepancy_measure1 = torch.zeros((len(testloader.dataset),)).cuda()
    discrepancy_measure2 = torch.zeros((len(testloader.dataset),)).cuda()
    agreement_measure = torch.zeros((len(testloader.dataset),))
    agreement_measure1 = torch.zeros((len(testloader.dataset),))

    # weighted k-nn correction,做了两次近邻标签修正
    print('starting the 1st knn....')
    with torch.no_grad():
        retrieval_one_hot_train = torch.zeros(k_val, class_num).cuda()
        for batch_idx, (_, _, index) in enumerate(testloader):
            batch_size = index.size(0)
            features = trainFeatures[index]

            # 当前batch的feature和其他所有样本的feature的cosine相似度距离
            dist = torch.mm(features, trainFeatures.t())  # B*N
            similiar_graph_all[index] = dist.cpu().detach()
            dist[torch.arange(dist.size(0)), index] = -1  # self-contrast set to -1

            # every sample have k-nearest neighbors
            yd, yi = dist.topk(k_val, dim=1, largest=True, sorted=True)  # yi:B*k, yd:B*K
            candidates = labels.view(1, -1).expand(batch_size, -1)  # replicate the labels #B*N
            retrieval = torch.gather(candidates, 1, yi)  # gather获取指定index的元素；获得topk的label
            # size(B*K);yi 是index; from replicated labels to get those of the topk neighbors using the index yi

            retrieval_one_hot_train.resize_(batch_size * k_val, class_num).zero_()  # B*k,C
            # generate the k*batchsize one-hot encoding from neighboring labels
            # set of neighbouring labels is turned into one-hot encoding
            # scatter_(dim, index, src),
            # dim=1, input[i][index[i][j]]的位置数值变为src[i][j]
            # retrieval.view后变为Bkc*1,
            retrieval_one_hot_train.scatter_(1, retrieval.view(-1, 1), 1)


            yd_transform = torch.exp(yd.clone().div_(cos_t))  # apply temperature to score
            # yd_transform[...] = 1.0  # to avoid using similarities

            # 通过k近邻改变当前样本的置信度, mul对应位相乘
            probs_corrected = torch.sum(torch.mul(retrieval_one_hot_train.view(batch_size, -1, class_num),
                                                  yd_transform.view(batch_size, -1, 1)), 1)
            # (B,K,C)*(B,K,1).sum=(B*C)，按照yd的程度加权近邻k的独热编码
            # 将数值归到（0，1）
            probs_norm = probs_corrected / torch.sum(probs_corrected, dim=1)[:, None]

            # 选出每个样本最高的类别相似度然后对过低or过高修正
            prob_temp = probs_norm[torch.arange(0, batch_size), labels[index]]
            prob_temp[prob_temp <= 1e-4] = 1e-4
            prob_temp[prob_temp > (1 - 1e-4)] = 1 - 1e-4
            discrepancy_measure1[index] = -torch.log(prob_temp)

            # 根据修正后的结果重新分配标签
            sorted_pro, predictions_corrected = probs_norm.sort(1, True)
            new_labels = predictions_corrected[:, 0]

            train_new_labels[index] = new_labels
            agreement_measure[index.data.cpu()] = (torch.max(probs_norm, dim=1)[1] == labels[index]).float().data.cpu()
        selected_examples = agreement_measure
    # 修正后的标签和原标签相似的做标记agreement_measure，说明该样本属于clean samples，属于集合T

    if two_knn or three_knn:
        print(f'starting knn2...')
        train_new_labels2 = train_new_labels.clone()  # N*1
        with torch.no_grad():
            retrieval_one_hot_train = torch.zeros(k_val, class_num).cuda()  #
            for batch_idx, (inputs, _, index) in enumerate(testloader):
                batchSize = inputs.size(0)
                features = trainFeatures[index]  # 64,256

                dist = torch.mm(features, trainFeatures.t())
                dist[torch.arange(dist.size()[0]), index] = -1  ##Self-contrast set to -1

                yd, yi = dist.topk(k_val, dim=1, largest=True, sorted=True)
                ## Top-K similar scores and corresponding indexes .yi:B*K
                candidates = train_new_labels2.view(1, -1).expand(batchSize, -1)  ##!!!这里变了 B*N
                retrieval = torch.gather(candidates, 1, yi)  # 对topk的近邻样本拿出来伪标签，B*K
                ## From replicated labels get those of the top-K neighbours using the index yi (from top-k operation)

                retrieval_one_hot_train.resize_(batchSize * k_val, class_num).zero_()  # B*k,C
                retrieval_one_hot_train.scatter_(1, retrieval.view(-1, 1), 1)  #

                yd_transform = torch.exp(yd.clone().div_(cos_t))  ## Apply temperature to scores
                # yd_transform[...] = 1.0  ##To avoid using similarities only counts
                probs_corrected = torch.sum(
                    torch.mul(retrieval_one_hot_train.view(batchSize, -1, class_num),
                              yd_transform.view(batchSize, -1, 1)), 1)
                # 修正的prob指的是周围k个的onehot乘以修正后的相似度(F*F.t)

                probs_norm = probs_corrected / torch.sum(probs_corrected, dim=1)[:, None]  # B*C


                prob_temp = probs_norm[torch.arange(0, batchSize), labels[index]]
                prob_temp[prob_temp <= 1e-4] = 1e-4
                prob_temp[prob_temp > (1 - 1e-4)] = 1 - 1e-4

                sorted_pro, predictions_corrected = probs_norm.sort(1, True)
                new_labels = predictions_corrected[:, 0]
                train_new_labels[index] = new_labels

                discrepancy_measure2[index] = -torch.log(prob_temp)

                agreement_measure1[index.data.cpu()] = (
                        torch.max(probs_norm, dim=1)[1] == labels[index]).float().data.cpu()
            selected_examples = agreement_measure1

    if three_knn:
        print(f'starting knn3...')
        train_new_labels2 = train_new_labels.clone()  # N*1
        with torch.no_grad():
            retrieval_one_hot_train = torch.zeros(k_val, class_num).cuda()  #
            for batch_idx, (inputs, _, index) in enumerate(testloader):
                batchSize = inputs.size(0)
                features = trainFeatures[index]  # 64,256

                dist = torch.mm(features, trainFeatures.t())
                dist[torch.arange(dist.size()[0]), index] = -1  ##Self-contrast set to -1

                yd, yi = dist.topk(k_val, dim=1, largest=True, sorted=True)
                ## Top-K similar scores and corresponding indexes .yi:B*K
                candidates = train_new_labels2.view(1, -1).expand(batchSize, -1)  ##!!!这里变了 B*N
                retrieval = torch.gather(candidates, 1, yi)  # 对topk的近邻样本拿出来伪标签，B*K
                ## From replicated labels get those of the top-K neighbours using the index yi (from top-k operation)

                retrieval_one_hot_train.resize_(batchSize * k_val, class_num).zero_()  # B*k,C
                retrieval_one_hot_train.scatter_(1, retrieval.view(-1, 1), 1)  #

                yd_transform = torch.exp(yd.clone().div_(cos_t))  ## Apply temperature to scores
                # yd_transform[...] = 1.0  ##To avoid using similarities only counts
                probs_corrected = torch.sum(
                    torch.mul(retrieval_one_hot_train.view(batchSize, -1, class_num),
                              yd_transform.view(batchSize, -1, 1)), 1)
                # 修正的prob指的是周围k个的onehot乘以修正后的相似度(F*F.t)

                probs_norm = probs_corrected / torch.sum(probs_corrected, dim=1)[:, None]  # B*C
                prob_temp = probs_norm[torch.arange(0, batchSize), labels[index]]
                prob_temp[prob_temp <= 1e-4] = 1e-4
                prob_temp[prob_temp > (1 - 1e-4)] = 1 - 1e-4

                discrepancy_measure2[index] = -torch.log(prob_temp)

                # prob_temp = probs_norm[torch.arange(0, batchSize), train_new_labels[index]]
                # prob_temp[prob_temp <= 1e-2] = 1e-2
                # prob_temp[prob_temp > (1 - 1e-2)] = 1 - 1e-2
                # discrepancy_measure2_pseudo_labels[index] = -torch.log(prob_temp)

                agreement_measure1[index.data.cpu()] = (
                        torch.max(probs_norm, dim=1)[1] == labels[index]).float().data.cpu()
            selected_examples = agreement_measure1
    # 修正后的标签和原标签相似的做标记agreement_measure，说明该样本属于clean samples，属于集合T
    # discrepancy_measure2_pseudo_labels = \hat y

    if two_knn or three_knn:
        discrepancy_measure = discrepancy_measure2
    else:
        discrepancy_measure = discrepancy_measure1

    # select examples
    if balance_class:
        num_clean_per_class = torch.zeros(class_num)
        for i in range(class_num):
            idx_class = labels == i
            num_clean_per_class[i] = torch.sum(selected_examples[idx_class])
        print(f'num_clean_per_class:{num_clean_per_class}')

        # the selected number of set G' is determined by the clean samples,
        # and think the sample with smaller discrepancy is the clean samples by default, but maybe not true.
        if (alpha == 0.5):  # 每类选出多少样本
            num_samples2select_class = torch.median(num_clean_per_class)
        elif (alpha == 1.0):
            num_samples2select_class = torch.max(num_clean_per_class)
        elif (alpha == 0.0):
            num_samples2select_class = torch.min(num_clean_per_class)
        else:
            num_samples2select_class = torch.quantile(num_clean_per_class, alpha, interpolation='nearest')
        # num_samples2select_class 决定了集合T
        agreement_measure = torch.zeros((len(labels),)).cuda()  # N,1
        print(f'num_samples2select_class:{num_samples2select_class}')
        for i in range(class_num):
            idx_class = labels == i  # N*1 noisy_beta标记label i的index
            samplesPerClass = idx_class.sum()
            # idx_class = torch.from_numpy(idx_class.astype("float"))
            idx_class = (idx_class.float() == 1.0).nonzero().squeeze().long()
            discrepancy_class = discrepancy_measure[idx_class]  # 根据周围的k个点以及

            if num_samples2select_class >= samplesPerClass:  # 如果该类数量少于阈值就全部选择
                k_corrected = samplesPerClass
            else:
                k_corrected = num_samples2select_class

            top_clean_class_relative_idx = \
                torch.topk(discrepancy_class, k=int(k_corrected), largest=False, sorted=False)[1].long()
            # 根据discrepancy_measure2，选出k_corrected个最小的index
            agreement_measure[idx_class[top_clean_class_relative_idx]] = 1.0  # 对于误差前k小的样本agreement标记为1，N*1
        selected_examples = agreement_measure  # 根据相似度选出的同一语义标签的样本index

    elif not balance_class and not sel_from_all:
        # first select clean samples that torch.argmax(prob_norm)==labels and then select topk as confident samples
        num_clean_per_class = torch.zeros(class_num)
        for i in range(class_num):
            idx_class = labels == i  # N*1 noisy_beta标记label i的index
            clean_num_i = torch.sum(selected_examples[idx_class])
            num_clean_per_class[i] = clean_num_i
        agreement_measure = torch.zeros((len(labels),)).cuda()  # N,1

        print(f'num_clean_per_class:{num_clean_per_class}')
        for i in range(class_num):
            idx_class = labels == i
            idx_class = (idx_class.float() == 1.0).float().nonzero().squeeze().long()
            discrepancy_class = discrepancy_measure[idx_class]  # 修正的距离分布

            k_corrected = sel_ratio * num_clean_per_class[i]
            if k_corrected > 0:
                top_clean_class_relative_idx = \
                    torch.topk(discrepancy_class, k=int(k_corrected), largest=False, sorted=False)[1].long()
                # 根据discrepancy_measure2，选出k_corrected个最小的index
                agreement_measure[idx_class[top_clean_class_relative_idx]] = 1.0  # 对于误差前k小的样本agreement标记为1，N*1
        selected_examples = agreement_measure  # 根据相似度选出的同一语义标签的样本index

    elif not balance_class and sel_from_all:
        # our sse setting
        # top gamma is top gamma of each class
        agreement_measure = torch.zeros((len(labels),)).cuda()  # N,1

        for i in range(class_num):
            idx_class = labels == i
            num_per_class = idx_class.sum()
            idx_class = (idx_class.float() == 1.0).float().nonzero().squeeze()
            discrepancy_class = discrepancy_measure[idx_class]  # 修正的距离分布

            k_corrected = sel_ratio * num_per_class
            if k_corrected >= 1:
                top_clean_class_relative_idx = \
                    torch.topk(discrepancy_class, k=int(k_corrected), largest=False, sorted=True)[1]
                # sorted means whether index is sorted accoring to their corresponding value
                # 根据discrepancy_measure2，选出k_corrected个最小的index
                i_sel_index = idx_class[top_clean_class_relative_idx]

                agreement_measure[i_sel_index] = 1.0  # 对于误差前k小的样本agreement标记为1，N*1
                if corrected:
                    '''
                    we found the confidence is dense scattered, we should also select samples with the score
                    that is same as the top selected
                    '''
                    the_val = discrepancy_class[top_clean_class_relative_idx[-1]]
                    the_val_index = (discrepancy_class == the_val).float().nonzero().squeeze().long()
                    agreement_measure[idx_class[the_val_index]] = 1.0

        selected_examples = agreement_measure
        # selected_examples is the variable to be consistent with the other if s
        # 根据相似度选出的同一语义标签的样本index
        # print(f'temp_record_selnum:{temp_record_selnum}')

    with torch.no_grad():
        index_selected = torch.nonzero(selected_examples, as_tuple=True)[0].cpu()  # N
        total_selected_num = len(index_selected)
        trainNoisyLabels = trainNoisyLabels.cpu().unsqueeze(1)  # N*1

        noisy_pairs = torch.eq(trainNoisyLabels, trainNoisyLabels.t())  # N*N if i,j have same label, [i,j]=1
        final_selected_pairs = torch.zeros_like(noisy_pairs).type(torch.uint8)

        # selected
        selected_pairs = noisy_pairs[
            index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num), index_selected.unsqueeze(
                0).expand(total_selected_num, total_selected_num)].clone()
        # temp_graph = similiar_graph_all[
        #     index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num), index_selected.unsqueeze(
        #         0).expand(total_selected_num, total_selected_num)]
        # selected_th = np.quantile(temp_graph[selected_pairs], beta)
        # #按照beta比例的分位数,beta是0则取最小值，beta=0时，final_selected_pairs是全部的, beta=1时，selected_pairs是selected_examples的对
        # temp = torch.zeros(total_num, total_num).type(torch.uint8)
        # 选出相似度高的，这些样本g''
        # noisy_pairs = torch.where(similiar_graph_all < selected_th, temp, noisy_pairs.type(torch.uint8)).type(torch.bool)#相似度小于指定比例beta就为0
        # #合并g' and g''
        # noisy_pairs[
        #     index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num), index_selected.unsqueeze(
        #         0).expand(total_selected_num, total_selected_num)] = selected_pairs#这个再放回去
        # final_selected_pairs = noisy_pairs
        final_selected_pairs[
            index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num), index_selected.unsqueeze(
                0).expand(total_selected_num, total_selected_num)] = selected_pairs.type(torch.uint8)
        final_selected_pairs = final_selected_pairs.type(torch.bool)
        # finally pairs that are selected and have same labels are set to true
    if plot:
        if two_knn:
            plt.hist(discrepancy_measure2.cpu().numpy(), color='blue', )
        else:
            plt.hist(discrepancy_measure1.cpu().numpy(), color='blue')
        plt.xlabel('score')
        plt.ylabel('freq')
        plt.title('freq/score')
        plt.savefig(f'./res/confidence/sse_confidence_epoch{epoch}.jpg')

    return selected_examples.cuda(), final_selected_pairs.contiguous()


def pair_selection_v1(k_val, testloader, labels, class_num, cos_t, knn_times, trainFeatures, balance_class=True,
                   sel_ratio=0, plot=False, epoch=-1, corrected=False):
    '''
    简化版本
    kval: neighbors number of knn
    label: the mem_label, pseudo-labels obtained by feature prototypes
    '''

    similiar_graph_all = torch.zeros(len(testloader.dataset), len(testloader.dataset))

    trainNoisyLabels = labels.clone().cuda()
    train_labels = labels.clone().cuda()
    discrepancy_measure = torch.zeros((len(testloader.dataset),)).cuda()
    agreement_measure = torch.zeros((len(testloader.dataset),))

    # weighted k-nn correction,做了两次近邻标签修正

    with torch.no_grad():
        for i in range(knn_times):
            print(f'starting the {i+1}st knn....')
            retrieval_one_hot_train = torch.zeros(k_val, class_num).cuda()
            train_new_labels = train_labels.clone()
            for batch_idx, (_, _, index) in enumerate(testloader):
                batch_size = index.size(0)
                features = trainFeatures[index]

                # 当前batch的feature和其他所有样本的feature的cosine相似度距离
                dist = torch.mm(features, trainFeatures.t())  # B*N
                similiar_graph_all[index] = dist.cpu().detach()
                dist[torch.arange(dist.size(0)), index] = -1  # self-contrast set to -1

                # every sample have k-nearest neighbors
                yd, yi = dist.topk(k_val, dim=1, largest=True, sorted=True)  # yi:B*k, yd:B*K
                candidates = train_new_labels.view(1, -1).expand(batch_size, -1)  # replicate the labels #B*N
                retrieval = torch.gather(candidates, 1, yi)  # gather获取指定index的元素；获得topk的label
                # size(B*K);yi 是index; from replicated labels to get those of the topk neighbors using the index yi

                retrieval_one_hot_train.resize_(batch_size * k_val, class_num).zero_()  # B*k,C
                # generate the k*batchsize one-hot encoding from neighboring labels
                # set of neighbouring labels is turned into one-hot encoding
                # scatter_(dim, index, src),
                # dim=1, input[i][index[i][j]]的位置数值变为src[i][j]
                # retrieval.view后变为Bkc*1,
                retrieval_one_hot_train.scatter_(1, retrieval.view(-1, 1), 1)


                yd_transform = torch.exp(yd.clone().div_(cos_t))  # apply temperature to score
                # yd_transform[...] = 1.0  # to avoid using similarities

                # 通过k近邻改变当前样本的置信度, mul对应位相乘
                probs_corrected = torch.sum(torch.mul(retrieval_one_hot_train.view(batch_size, -1, class_num),
                                                      yd_transform.view(batch_size, -1, 1)), 1)
                # (B,K,C)*(B,K,1).sum=(B*C)，按照yd的程度加权近邻k的独热编码
                # 将数值归到（0，1）
                probs_norm = probs_corrected / torch.sum(probs_corrected, dim=1)[:, None]

                # 选出每个样本对于最初始的pseudo labels的prob_norm
                prob_temp = probs_norm[torch.arange(0, batch_size), labels[index]]
                prob_temp[prob_temp <= 1e-4] = 1e-4
                prob_temp[prob_temp > (1 - 1e-4)] = 1 - 1e-4
                discrepancy_measure[index] = -torch.log(prob_temp)

                # 根据修正后的结果重新分配标签
                sorted_pro, predictions_corrected = probs_norm.sort(1, True)
                new_labels = predictions_corrected[:, 0]

                train_labels[index] = new_labels
                agreement_measure[index.data.cpu()] = (torch.max(probs_norm, dim=1)[1] == labels[index]).float().data.cpu()
            selected_examples = agreement_measure


    if balance_class:
        # balance_class: different from previous versions, here it means: whether to select the same gamma ratio from each class
        # our sse setting
        # top gamma is top gamma of each class
        agreement_measure = torch.zeros((len(labels),)).cuda()  # N,1

        for i in range(class_num):
            idx_class = labels == i
            num_per_class = idx_class.sum()
            idx_class = (idx_class.float() == 1.0).float().nonzero().squeeze()
            discrepancy_class = discrepancy_measure[idx_class]

            k_corrected = sel_ratio * num_per_class
            if k_corrected >= 1:
                top_clean_class_relative_idx = \
                    torch.topk(discrepancy_class, k=int(k_corrected), largest=False, sorted=True)[1]

                # 选出k_corrected个最小的index
                i_sel_index = idx_class[top_clean_class_relative_idx]
                # 对于误差前k小的样本agreement标记为1，N*1
                agreement_measure[i_sel_index] = 1.0
                if corrected:
                    '''
                    we found the confidence is dense scattered, we should also select samples with the score
                    that is same as the top selected
                    '''
                    the_val = discrepancy_class[top_clean_class_relative_idx[-1]]
                    the_val_index = (discrepancy_class == the_val).float().nonzero().squeeze().long()
                    agreement_measure[idx_class[the_val_index]] = 1.0

        selected_examples = agreement_measure
        # selected_examples is the variable to be consistent with the other if s
        # 根据相似度选出的同一语义标签的样本index
        # print(f'temp_record_selnum:{temp_record_selnum}')

    with torch.no_grad():
        index_selected = torch.nonzero(selected_examples, as_tuple=True)[0].cpu()  # N
        total_selected_num = len(index_selected)
        trainNoisyLabels = trainNoisyLabels.cpu().unsqueeze(1)  # N*1

        noisy_pairs = torch.eq(trainNoisyLabels, trainNoisyLabels.t())  # N*N if i,j have same label, [i,j]=1
        final_selected_pairs = torch.zeros_like(noisy_pairs).type(torch.uint8)

        # selected
        selected_pairs = noisy_pairs[
            index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num), index_selected.unsqueeze(
                0).expand(total_selected_num, total_selected_num)].clone()
        final_selected_pairs[
            index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num), index_selected.unsqueeze(
                0).expand(total_selected_num, total_selected_num)] = selected_pairs.type(torch.uint8)
        final_selected_pairs = final_selected_pairs.type(torch.bool)
        # finally pairs that are selected and have same labels are set to true
    if plot:
        plt.hist(discrepancy_measure.cpu().numpy(), color='blue', )
        plt.xlabel('score')
        plt.ylabel('freq')
        plt.title('freq/score')
        plt.savefig(f'./res/confidence/sse_confidence_epoch{epoch}.jpg')

    return selected_examples.cuda(), final_selected_pairs.contiguous()



def pair_selection_by_p(testloader, labels, class_num, all_outputs, sel_ratio=0):
    # PL + maxp,选择前ratio的
    # ssl + maxp(ssl_p)
    # both are p in pseudo-label index,
    trainNoisyLabels = labels.clone().cuda()
    discrepancy_measure = torch.zeros((len(testloader.dataset.imgs),)).cuda()

    print(f'starting select topk p ...')
    with torch.no_grad():
        for batch_idx, (_, _, index) in enumerate(testloader):
            batchSize = index.size(0)

            output_p = all_outputs[index]
            output_p = output_p[torch.arange(0, batchSize), labels[index]]
            discrepancy_measure[index] = output_p

    agreement_measure = torch.zeros((len(labels),)).cuda()  # N,1
    for i in range(class_num):
        idx_class = labels == i
        num_per_class = idx_class.sum()
        idx_class = (idx_class.float() == 1.0).nonzero().squeeze().long()

        discrepancy_class = discrepancy_measure[idx_class]  # 修正的距离分布

        k_corrected = sel_ratio * num_per_class
        if k_corrected > 0:
            top_clean_class_relative_idx = \
                torch.topk(discrepancy_class, k=int(k_corrected), largest=True, sorted=False)[1].long()
            # 根据discrepancy_measure2，选出k_corrected个最小的index
            agreement_measure[idx_class[top_clean_class_relative_idx].long()] = 1.0  # 对于误差前k小的样本agreement标记为1，N*1
    selected_examples = agreement_measure  # 根据相似度选出的同一语义标签的样本index

    with torch.no_grad():
        index_selected = torch.nonzero(selected_examples, as_tuple=True)[0].cpu()  # N
        total_selected_num = len(index_selected)
        trainNoisyLabels = trainNoisyLabels.cpu().unsqueeze(1)  # N*1
        # total_num = len(trainNoisyLabels)
        noisy_pairs = torch.eq(trainNoisyLabels, trainNoisyLabels.t())  # N*N
        final_selected_pairs = torch.zeros_like(noisy_pairs)

        # 暂存选择的同类clean samples
        selected_pairs = noisy_pairs[
            index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num), index_selected.unsqueeze(
                0).expand(total_selected_num, total_selected_num)].clone()
        final_selected_pairs[
            index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num), index_selected.unsqueeze(
                0).expand(total_selected_num, total_selected_num)] = selected_pairs  # 这个再放回去

    return selected_examples.cuda(), final_selected_pairs.contiguous()


def pair_selection_by_ent(testloader, labels, class_num, all_outputs, sel_ratio=0):
    # PL + ent
    # SSPL + ent
    # depending on the pseudo_label
    trainNoisyLabels = labels.clone().cuda()
    discrepancy_measure = torch.zeros((len(testloader.dataset.imgs),)).cuda()

    print(f'starting select topk p ...')
    with torch.no_grad():
        for batch_idx, (_, _, index) in enumerate(testloader):
            output_p = all_outputs[index]
            # a bug ! error from CVPR paper
            # output_p = output_p[torch.arange(0, batchSize), labels[index]]
            ent = Loss.Entropy(output_p)
            discrepancy_measure[index] = ent

    agreement_measure = torch.zeros((len(labels),)).cuda()  # N,1
    for i in range(class_num):
        idx_class = labels == i
        num_per_class = idx_class.sum()
        idx_class = (idx_class.float() == 1.0).nonzero().squeeze().long()

        discrepancy_class = discrepancy_measure[idx_class]  # 修正的距离分布
        k_corrected = sel_ratio * num_per_class
        if k_corrected > 0:
            top_clean_class_relative_idx = \
                torch.topk(discrepancy_class, k=int(k_corrected), largest=False, sorted=False)[1].long()
            # 根据discrepancy_measure2，选出k_corrected个最小的index
            agreement_measure[idx_class[top_clean_class_relative_idx].long()] = 1.0  # 对于误差前k小的样本agreement标记为1，N*1
    selected_examples = agreement_measure  # 根据相似度选出的同一语义标签的样本index

    with torch.no_grad():
        index_selected = torch.nonzero(selected_examples, as_tuple=True)[0].cpu()  # N
        total_selected_num = len(index_selected)
        trainNoisyLabels = trainNoisyLabels.cpu().unsqueeze(1)  # N*1
        # total_num = len(trainNoisyLabels)
        noisy_pairs = torch.eq(trainNoisyLabels, trainNoisyLabels.t())  # N*N
        final_selected_pairs = torch.zeros_like(noisy_pairs)

        # 暂存选择的同类clean samples
        selected_pairs = noisy_pairs[
            index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num), index_selected.unsqueeze(
                0).expand(total_selected_num, total_selected_num)].clone()
        final_selected_pairs[
            index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num), index_selected.unsqueeze(
                0).expand(total_selected_num, total_selected_num)] = selected_pairs  # 这个再放回去

    return selected_examples.cuda(), final_selected_pairs.contiguous()


def pair_selection_by_p_thresh(testloader, labels, all_outputs, thresh=0.5):
    # maxp; select samples that have p > thresh
    trainNoisyLabels = labels.clone().cuda()
    discrepancy_measure = torch.zeros((len(testloader.dataset.imgs),)).cuda()

    print(f'starting select topk p ...')
    with torch.no_grad():
        for batch_idx, (inputs, _, index) in enumerate(testloader):
            batchSize = inputs.size(0)
            output_p = all_outputs[index]  # 64,c
            temp_p = output_p[torch.arange(0, batchSize), labels[index]]  # size 50
            discrepancy_measure[index] = temp_p

    agreement_measure = torch.zeros((len(labels),)).cuda()  # N,1
    # if the threshold is set, we dont need to delve into every class

    top_clean_class_relative_idx = torch.nonzero((discrepancy_measure > thresh).float()).squeeze()
    agreement_measure[top_clean_class_relative_idx] = 1.0  # 对于误差前k小的样本agreement标记为1，N*1

    selected_examples = agreement_measure  # 根据相似度选出的同一语义标签的样本index
    print(f'the ratio of selected samples:{selected_examples.data.sum() / len(labels)}')

    with torch.no_grad():
        index_selected = torch.nonzero(selected_examples, as_tuple=True)[0].cpu()  # N
        total_selected_num = len(index_selected)
        trainNoisyLabels = trainNoisyLabels.cpu().unsqueeze(1)  # N*1
        noisy_pairs = torch.eq(trainNoisyLabels, trainNoisyLabels.t())  # N*N is torch.eye
        final_selected_pairs = torch.zeros_like(noisy_pairs)

        # 暂存选择的同类clean samples
        selected_pairs = noisy_pairs[
            index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num), index_selected.unsqueeze(
                0).expand(total_selected_num, total_selected_num)].clone()
        final_selected_pairs[
            index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num), index_selected.unsqueeze(
                0).expand(total_selected_num, total_selected_num)] = selected_pairs  # 这个再放回去

    return selected_examples.cuda(), final_selected_pairs.contiguous()


def pair_selection_low(k_val, testloader, labels, class_num, cos_t, trainFeatures, sel_ratio=0, sel_from_all=True,
                       two_knn=True):
    # 选出低置信度的是correction label和原来不一致的
    trainFeatures = trainFeatures.t()
    similiar_graph_all = torch.zeros(len(testloader.dataset), len(testloader.dataset))

    trainNoisyLabels = labels.clone().cuda()
    train_new_labels = labels.clone().cuda()
    discrepancy_measure1 = torch.zeros((len(testloader.dataset.imgs),)).cuda()
    discrepancy_measure2 = torch.zeros((len(testloader.dataset.imgs),)).cuda()
    agreement_measure = torch.zeros((len(testloader.dataset.imgs),))
    disagreement_measure = torch.zeros((len(testloader.dataset.imgs),))

    # weighted k-nn correction,做了两次近邻标签修正
    print('starting knn1....')
    with torch.no_grad():
        retrieval_one_hot_train = torch.zeros(k_val, class_num).cuda()
        for batch_idx, (inputs, _, index) in enumerate(testloader):
            batch_size = inputs.size(0)
            features = trainFeatures.t()[index]

            dist = torch.mm(features, trainFeatures)  # B*N
            similiar_graph_all[index] = dist.cpu().detach()
            dist[torch.arange(dist.size(0)), index] = -1  # self-contrast set to -1

            yd, yi = dist.topk(k_val, dim=1, largest=True, sorted=True)  # yi:B*K
            candidates = labels.view(1, -1).expand(batch_size, -1)  # replicate the labels #B*N
            retrieval = torch.gather(candidates, 1, yi)  # gather获取指定index的元素；获得topk的label
            # size(B*K);yi 是index; from replicated labels to get those of the topk neighbors using the index yi

            retrieval_one_hot_train.resize_(batch_size * k_val, class_num).zero_()  # B*k,C
            # generate the k*batchsize one-hot encoding from neighboring labels
            # set of neighbouring labels is turned into one-hot encoding
            retrieval_one_hot_train.scatter_(1, retrieval.view(-1, 1), 1)  # scatter改变自身指定位置元素，变为one-hot向量

            yd_transform = torch.exp(yd.clone().div_(cos_t))  # apply temperature to score
            # yd_transform[...] = 1.0  # to avoid using similarities

            # 通过k近邻改变当前样本的
            probs_corrected = torch.sum(torch.mul(retrieval_one_hot_train.view(batch_size, -1, class_num),
                                                  yd_transform.view(batch_size, -1, 1)), 1)
            # (B,K,C)*(B,K,1).sum=(B*C)，按照yd的程度加权近邻k的独热编码
            probs_norm = probs_corrected / torch.sum(probs_corrected, dim=1)[:, None]

            # 选出每个样本最高的类别相似度然后对过低or过高修正
            prob_temp = probs_norm[torch.arange(0, batch_size), labels[index]]
            prob_temp[prob_temp <= 1e-2] = 1e-2
            prob_temp[prob_temp > (1 - 1e-2)] = 1 - 1e-2
            discrepancy_measure1[index] = -torch.log(prob_temp)

            # 根据修正后的重新分类标签
            sorted_pro, predictions_corrected = probs_norm.sort(1, True)
            new_labels = predictions_corrected[:, 0]

            train_new_labels[index] = new_labels
            agreement_measure[index.data.cpu()] = (torch.max(probs_norm, dim=1)[1] == labels[index]).float().data.cpu()
            disagreement_measure[index.data.cpu()] = (
                    torch.max(probs_norm, dim=1)[1] != labels[index]).float().data.cpu()
    # 修正后的标签和原标签相似的做标记agreement_measure，说明该样本属于clean samples，属于集合T

    # train_new_labels=\widetilde y
    if two_knn:
        print(f'starting knn2...')
        train_new_labels2 = train_new_labels.clone()  # N*1
        with torch.no_grad():
            retrieval_one_hot_train = torch.zeros(k_val, class_num).cuda()  #
            for batch_idx, (inputs, _, index) in enumerate(testloader):
                batchSize = inputs.size(0)
                features = trainFeatures.t()[index]  # 64,256

                dist = torch.mm(features, trainFeatures)
                dist[torch.arange(dist.size()[0]), index] = -1  ##Self-contrast set to -1

                yd, yi = dist.topk(k_val, dim=1, largest=True,
                                   sorted=True)  ## Top-K similar scores and corresponding indexes .yi:B*K
                candidates = train_new_labels2.view(1, -1).expand(batchSize, -1)  ##!!!这里变了 B*N
                retrieval = torch.gather(candidates, 1, yi)  # 对于topk的近邻样本选出来，B*K
                ## From replicated labels get those of the top-K neighbours using the index yi (from top-k operation)

                retrieval_one_hot_train.resize_(batchSize * k_val, class_num).zero_()  # B*k,C
                retrieval_one_hot_train.scatter_(1, retrieval.view(-1, 1), 1)  #
                yd_transform = torch.exp(yd.clone().div_(cos_t))  ## Apply temperature to scores
                # yd_transform[...] = 1.0  ##To avoid using similarities only counts
                probs_corrected = torch.sum(
                    torch.mul(retrieval_one_hot_train.view(batchSize, -1, class_num),
                              yd_transform.view(batchSize, -1, 1)), 1)
                # 修正的prob指的是周围k个的onehot乘以修正后的相似度(F*F.t)

                probs_norm = probs_corrected / torch.sum(probs_corrected, dim=1)[:, None]  # B*C

                prob_temp = probs_norm[torch.arange(0, batchSize), labels[index]]
                prob_temp[prob_temp <= 1e-2] = 1e-2
                prob_temp[prob_temp > (1 - 1e-2)] = 1 - 1e-2

                discrepancy_measure2[index] = -torch.log(prob_temp)

                agreement_measure[index.data.cpu()] = (
                        torch.max(probs_norm, dim=1)[1] == labels[index]).float().data.cpu()
                disagreement_measure[index.data.cpu()] = (
                        torch.max(probs_norm, dim=1)[1] != labels[index]).float().data.cpu()
    print(f'disareement sum number:{disagreement_measure.float().sum().item()}')
    # 修正后的标签和原标签相似的做标记agreement_measure，说明该样本属于clean samples，属于集合T
    if not sel_from_all:
        num_clean_per_class = torch.zeros(class_num)
        for i in range(class_num):
            idx_class = labels == i  # N*1 noisy_beta标记label i的index
            clean_num_i = torch.sum(agreement_measure[idx_class])
            num_clean_per_class[i] = clean_num_i
        agreement_measure = torch.zeros((len(labels),)).cuda()  # N,1

        print(f'num_clean_per_class:{num_clean_per_class}')
        for i in range(class_num):
            idx_class = labels == i
            idx_class = (idx_class.float() == 1.0).nonzero().squeeze().long()
            discrepancy_class = discrepancy_measure2[idx_class]  # 修正的距离分布

            k_corrected = sel_ratio * num_clean_per_class[i]
            if k_corrected > 0:
                top_clean_class_relative_idx = \
                    torch.topk(discrepancy_class, k=int(k_corrected), largest=False, sorted=False)[1].long()
                # 根据discrepancy_measure2，选出k_corrected个最小的index
                agreement_measure[idx_class[top_clean_class_relative_idx]] = 1.0  # 对于误差前k小的样本agreement标记为1，N*1
        selected_examples = agreement_measure  # 根据相似度选出的同一语义标签的样本index

    elif sel_from_all:
        agreement_measure = torch.zeros((len(labels),)).cuda()  # N,1
        for i in range(class_num):
            idx_class = labels == i
            num_per_class = idx_class.sum()
            idx_class = (idx_class.float() == 1.0).nonzero().squeeze().long()
            if two_knn:
                discrepancy_class = discrepancy_measure2[idx_class]  # 修正的距离分布
            else:
                discrepancy_class = discrepancy_measure1[idx_class]  # 修正的距离分布
            k_corrected = sel_ratio * num_per_class
            if k_corrected > 0:
                # sel_ratio=1 代表选择这个类别(按照label)数量的top个近邻样本，这些样本很大概率不都是属于这一类。
                top_clean_class_relative_idx = \
                    torch.topk(discrepancy_class, k=int(k_corrected), largest=False, sorted=False)[1].long()
                # 根据discrepancy_measure2，选出k_corrected个最小的index
                agreement_measure[idx_class[top_clean_class_relative_idx].long()] = 1.0  # 对于误差前k小的样本agreement标记为1，N*1
        selected_examples = agreement_measure  # 根据相似度选出的同一语义标签的样本index

    with torch.no_grad():
        index_selected = torch.nonzero(selected_examples, as_tuple=True)[0].cpu()  # N
        total_selected_num = len(index_selected)
        trainNoisyLabels = trainNoisyLabels.cpu().unsqueeze(1)  # N*1
        noisy_pairs = torch.eq(trainNoisyLabels, trainNoisyLabels.t())  # N*N
        final_selected_pairs = torch.zeros_like(noisy_pairs).type(torch.uint8)

        # #暂存选择的同类clean samples
        selected_pairs = noisy_pairs[
            index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num), index_selected.unsqueeze(
                0).expand(total_selected_num, total_selected_num)].clone()

        final_selected_pairs[
            index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num), index_selected.unsqueeze(
                0).expand(total_selected_num, total_selected_num)] = selected_pairs.type(torch.uint8)
        final_selected_pairs = final_selected_pairs.type(torch.bool)
    return selected_examples.cuda(), final_selected_pairs.contiguous(), disagreement_measure.cuda()


def pair_selection_thresh(k_val, testloader, labels, class_num, cos_t, trainFeatures, thresh=0.3, two_knn=True):
    # 根据估计的概率阈值选择confident samples
    # log0.5~=-0.3,所以threshold默认0.3代表当前类的估计概率应该在0.5以上才能是confident samples
    trainFeatures = trainFeatures.t()
    similiar_graph_all = torch.zeros(len(testloader.dataset), len(testloader.dataset))

    trainNoisyLabels = labels.clone().cuda()
    train_new_labels = labels.clone().cuda()
    discrepancy_measure1 = torch.zeros((len(testloader.dataset.imgs),)).cuda()
    discrepancy_measure2 = torch.zeros((len(testloader.dataset.imgs),)).cuda()
    agreement_measure = torch.zeros((len(testloader.dataset.imgs),))

    # weighted k-nn correction,做了两次近邻标签修正
    print('starting knn1....')
    with torch.no_grad():
        retrieval_one_hot_train = torch.zeros(k_val, class_num).cuda()
        for batch_idx, (inputs, _, index) in enumerate(testloader):
            batch_size = inputs.size(0)
            features = trainFeatures.t()[index]

            dist = torch.mm(features, trainFeatures)  # B*N
            similiar_graph_all[index] = dist.cpu().detach()
            dist[torch.arange(dist.size(0)), index] = -1  # self-contrast set to -1

            yd, yi = dist.topk(k_val, dim=1, largest=True, sorted=True)  # yi:B*K
            candidates = labels.view(1, -1).expand(batch_size, -1)  # replicate the labels #B*N
            retrieval = torch.gather(candidates, 1, yi)  # gather获取指定index的元素；获得topk的label
            # size(B*K);yi 是index; from replicated labels to get those of the topk neighbors using the index yi

            retrieval_one_hot_train.resize_(batch_size * k_val, class_num).zero_()  # B*k,C
            # generate the k*batchsize one-hot encoding from neighboring labels
            # set of neighbouring labels is turned into one-hot encoding
            retrieval_one_hot_train.scatter_(1, retrieval.view(-1, 1), 1)  # scatter改变自身指定位置元素，变为one-hot向量

            yd_transform = torch.exp(yd.clone().div_(cos_t))  # apply temperature to score
            # yd_transform[...] = 1.0  # to avoid using similarities

            # 通过k近邻改变当前样本的
            probs_corrected = torch.sum(torch.mul(retrieval_one_hot_train.view(batch_size, -1, class_num),
                                                  yd_transform.view(batch_size, -1, 1)), 1)
            # (B,K,C)*(B,K,1).sum=(B*C)，按照yd的程度加权近邻k的独热编码
            probs_norm = probs_corrected / torch.sum(probs_corrected, dim=1)[:, None]

            # 选出每个样本最高的类别相似度然后对过低or过高修正
            prob_temp = probs_norm[torch.arange(0, batch_size), labels[index]]
            prob_temp[prob_temp <= 1e-2] = 1e-2
            prob_temp[prob_temp > (1 - 1e-2)] = 1 - 1e-2
            discrepancy_measure1[index] = prob_temp

            # 根据修正后的重新分类标签
            sorted_pro, predictions_corrected = probs_norm.sort(1, True)
            new_labels = predictions_corrected[:, 0]

            train_new_labels[index] = new_labels
            agreement_measure[index.data.cpu()] = (torch.max(probs_norm, dim=1)[1] == labels[index]).float().data.cpu()

    # 修正后的标签和原标签相似的做标记agreement_measure，说明该样本属于clean samples，属于集合T

    # train_new_labels=\widetilde y
    if two_knn:
        print(f'starting knn2...')
        train_new_labels2 = train_new_labels.clone()  # N*1
        with torch.no_grad():
            retrieval_one_hot_train = torch.zeros(k_val, class_num).cuda()  #
            for batch_idx, (inputs, _, index) in enumerate(testloader):
                batchSize = inputs.size(0)
                features = trainFeatures.t()[index]  # 64,256

                dist = torch.mm(features, trainFeatures)
                dist[torch.arange(dist.size()[0]), index] = -1  ##Self-contrast set to -1

                yd, yi = dist.topk(k_val, dim=1, largest=True, sorted=True)
                ## Top-K similar scores and corresponding indexes .yi:B*K
                candidates = train_new_labels2.view(1, -1).expand(batchSize, -1)  ##!!!这里变了 B*N
                retrieval = torch.gather(candidates, 1, yi)  # 对于topk的近邻样本选出来，B*K
                ## From replicated labels get those of the top-K neighbours using the index yi (from top-k operation)

                retrieval_one_hot_train.resize_(batchSize * k_val, class_num).zero_()  # B*k,C
                retrieval_one_hot_train.scatter_(1, retrieval.view(-1, 1), 1)  #
                yd_transform = torch.exp(yd.clone().div_(cos_t))  ## Apply temperature to scores
                # yd_transform[...] = 1.0  ##To avoid using similarities only counts
                probs_corrected = torch.sum(
                    torch.mul(retrieval_one_hot_train.view(batchSize, -1, class_num),
                              yd_transform.view(batchSize, -1, 1)), 1)
                # 修正的prob指的是周围k个的onehot乘以修正后的相似度(F*F.t)

                probs_norm = probs_corrected / torch.sum(probs_corrected, dim=1)[:, None]  # B*C

                prob_temp = probs_norm[torch.arange(0, batchSize), labels[index]]
                prob_temp[prob_temp <= 1e-2] = 1e-2
                prob_temp[prob_temp > (1 - 1e-2)] = 1 - 1e-2
                discrepancy_measure2[index] = prob_temp
                # agreement_measure[index.data.cpu()] = (torch.max(probs_norm, dim=1)[1] == labels[index]).float().data.cpu()

    # 修正后的标签和原标签相似的做标记agreement_measure，说明该样本属于clean samples，属于集合T
    agreement_measure = torch.zeros((len(labels),)).cuda()  # N,1
    if two_knn:
        discrepancy_measure = discrepancy_measure2
    else:
        discrepancy_measure = discrepancy_measure1
    top_clean_class_relative_idx = torch.nonzero((discrepancy_measure > thresh).float()).squeeze().long()
    agreement_measure[top_clean_class_relative_idx] = 1.0  # 对于误差前k小的样本agreement标记为1，N*1
    selected_examples = agreement_measure  # 根据相似度选出的同一语义标签的样本index

    print(f'number of selected samples:{selected_examples.data.sum()}')

    with torch.no_grad():
        index_selected = torch.nonzero(selected_examples, as_tuple=True)[0].cpu()  # N
        total_selected_num = len(index_selected)
        trainNoisyLabels = trainNoisyLabels.cpu().unsqueeze(1)  # N*1
        noisy_pairs = torch.eq(trainNoisyLabels, trainNoisyLabels.t())  # N*N
        final_selected_pairs = torch.zeros_like(noisy_pairs).type(torch.uint8)

        # #暂存选择的同类clean samples
        selected_pairs = noisy_pairs[
            index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num), index_selected.unsqueeze(
                0).expand(total_selected_num, total_selected_num)].clone()

        final_selected_pairs[
            index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num), index_selected.unsqueeze(
                0).expand(total_selected_num, total_selected_num)] = selected_pairs.type(torch.uint8)
        final_selected_pairs = final_selected_pairs.type(torch.bool)
    return selected_examples.cuda(), final_selected_pairs.contiguous()

def pair_selection_by_center_ratio(testloader, labels, class_num, beta, trainFeatures, initc, sel_ratio):
    # sspl + cossim
    dd = cdist(trainFeatures.cpu().numpy(), initc.cpu().numpy(), 'cosine')  # N*C
    dd = torch.from_numpy(dd).cuda()
    agreement_measure = torch.zeros((len(testloader.dataset),)).cuda()

    dd_center = dd[torch.arange(0, dd.size(0)), labels]  # N  距离聚类中心的距离
    for i in range(class_num):
        class_index = labels == i  # N
        class_num = torch.sum(class_index)
        class_index = torch.nonzero(class_index).squeeze()  # n_c 该类比在所有data中的index

        class_dd = dd_center[class_index]  # 这里已经将原来的index弄丢了#index

        class_sel = class_num * sel_ratio
        yd, yi = class_dd.topk(int(class_sel), largest=False, sorted=True)
        sel_index = class_index[yi]
        agreement_measure[sel_index] = 1.

    trainNoisyLabels = labels.clone()
    selected_examples = agreement_measure  # 根据相似度选出的同一语义标签的样本index
    print(f'selected_examples.shape:{selected_examples.shape}')
    with torch.no_grad():
        index_selected = torch.nonzero(selected_examples, as_tuple=True)[0].cpu()  # N
        total_selected_num = len(index_selected)
        trainNoisyLabels = trainNoisyLabels.cpu().unsqueeze(1)  # N*1
        noisy_pairs = torch.eq(trainNoisyLabels, trainNoisyLabels.t())  # N*N
        final_selected_pairs = torch.zeros_like(noisy_pairs)

        # 暂存选择的同类clean samples
        selected_pairs = noisy_pairs[
            index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num), index_selected.unsqueeze(
                0).expand(total_selected_num, total_selected_num)].clone()
        final_selected_pairs[
            index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num), index_selected.unsqueeze(
                0).expand(total_selected_num, total_selected_num)] = selected_pairs  # 这个再放回去

    return selected_examples, final_selected_pairs.contiguous()


def pair_selection_JMDS(labels, JMDS, class_num, sel_ratio=0, ):
    agreement_measure = torch.zeros((len(labels),)).cuda()  # N,1

    for i in range(class_num):
        idx_class = labels == i
        num_per_class = idx_class.sum()
        idx_class = (idx_class.float() == 1.0).float().nonzero().squeeze()

        discrepancy_class = JMDS[idx_class]  # 修正的距离分布
        k_corrected = sel_ratio * num_per_class
        if k_corrected > 0:
            top_clean_class_relative_idx = \
                torch.topk(discrepancy_class, k=int(k_corrected), largest=True, sorted=False)[1]
            # 根据discrepancy_measure2，选出k_corrected个最小的index
            i_sel_index = idx_class[top_clean_class_relative_idx].long()
            # temp_record_selnum[i]=i_sel_index.size(0)

            agreement_measure[i_sel_index] = 1.0  # 对于误差前k小的样本agreement标记为1，N*1
    selected_examples = agreement_measure  # 根据相似度选出的同一语义标签的样本index

    with torch.no_grad():
        index_selected = torch.nonzero(selected_examples, as_tuple=True)[0].cpu()  # N
        total_selected_num = len(index_selected)
        trainNoisyLabels = labels.cpu().unsqueeze(1)  # N*1
        # total_num = len(trainNoisyLabels)
        noisy_pairs = torch.eq(trainNoisyLabels, trainNoisyLabels.t())  # N*N
        final_selected_pairs = torch.zeros_like(noisy_pairs).type(torch.uint8)

        # #暂存选择的同类clean samples
        selected_pairs = noisy_pairs[
            index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num), index_selected.unsqueeze(
                0).expand(total_selected_num, total_selected_num)].clone()
        final_selected_pairs[
            index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num), index_selected.unsqueeze(
                0).expand(total_selected_num, total_selected_num)] = selected_pairs.type(torch.uint8)
        final_selected_pairs = final_selected_pairs.type(torch.bool)

    return selected_examples.cuda(), final_selected_pairs.contiguous()


def pair_selection_01(k_val, testloader, labels, class_num, cos_t, trainFeatures,
                      sel_ratio=0, two_knn=True, plot=False, epoch=-1, corrected=False):
    # retain the discrepancy between [0,1] to give more choice for sample selection
    similiar_graph_all = torch.zeros(len(testloader.dataset), len(testloader.dataset))

    trainNoisyLabels = labels.clone().cuda()
    train_new_labels = labels.clone().cuda()
    discrepancy_measure1 = torch.zeros((len(testloader.dataset),)).cuda()
    discrepancy_measure2 = torch.zeros((len(testloader.dataset),)).cuda()

    agreement_measure2 = torch.zeros((len(labels, )))
    # weighted k-nn correction,做了两次近邻标签修正
    print('starting knn1....')
    with torch.no_grad():
        retrieval_one_hot_train = torch.zeros(k_val, class_num).cuda()
        for batch_idx, (inputs, _, index) in enumerate(testloader):
            batch_size = inputs.size(0)
            features = trainFeatures[index]

            dist = torch.mm(features, trainFeatures.t())  # B*N
            similiar_graph_all[index] = dist.cpu().detach()
            dist[torch.arange(dist.size(0)), index] = -1  # self-contrast set to -1

            yd, yi = dist.topk(k_val, dim=1, largest=True, sorted=True)  # yi:B*K
            candidates = labels.view(1, -1).expand(batch_size, -1)  # replicate the labels #B*N
            retrieval = torch.gather(candidates, 1, yi)  # gather获取指定index的元素；获得topk的label
            # size(B*K);yi 是index; from replicated labels to get those of the topk neighbors using the index yi

            retrieval_one_hot_train.resize_(batch_size * k_val, class_num).zero_()  # B*k,C
            # generate the k*batchsize one-hot encoding from neighboring labels
            # set of neighbouring labels is turned into one-hot encoding
            retrieval_one_hot_train.scatter_(1, retrieval.view(-1, 1), 1)  # scatter改变自身指定位置元素，变为one-hot向量

            yd_transform = torch.exp(yd.clone().div_(cos_t))  # apply temperature to score
            # yd_transform[...] = 1.0  # to avoid using similarities

            # 通过k近邻改变当前样本的
            probs_corrected = torch.sum(torch.mul(retrieval_one_hot_train.view(batch_size, -1, class_num),
                                                  yd_transform.view(batch_size, -1, 1)), 1)
            # (B,K,C)*(B,K,1).sum=(B*C)，按照yd的程度加权近邻k的独热编码
            probs_norm = probs_corrected / torch.sum(probs_corrected, dim=1)[:, None]

            # 选出每个样本最高的类别相似度然后对过低or过高修正
            prob_temp = probs_norm[torch.arange(0, batch_size), labels[index]]
            discrepancy_measure1[index] = prob_temp

            # 根据修正后的重新分类标签
            sorted_pro, predictions_corrected = probs_norm.sort(1, True)  # descending = true
            new_labels = predictions_corrected[:, 0]

            train_new_labels[index] = new_labels
    # 修正后的标签和原标签相似的做标记agreement_measure，说明该样本属于clean samples，属于集合T

    if two_knn:
        print(f'starting knn2...')
        train_new_labels2 = train_new_labels.clone()  # N*1
        with torch.no_grad():
            retrieval_one_hot_train = torch.zeros(k_val, class_num).cuda()  #
            for batch_idx, (inputs, _, index) in enumerate(testloader):
                batchSize = inputs.size(0)
                features = trainFeatures[index]  # 64,256

                batch_label = labels[index]
                dist = torch.mm(features, trainFeatures.t())
                dist[torch.arange(dist.size()[0]), index] = -1  ##Self-contrast set to -1

                yd, yi = dist.topk(k_val, dim=1, largest=True,
                                   sorted=True)
                ## Top-K similar scores and corresponding indexes .yi:B*K
                candidates = train_new_labels2.view(1, -1).expand(batchSize, -1)  ##!!!这里变了 B*N
                retrieval = torch.gather(candidates, 1, yi)  # 对topk的近邻样本拿出来伪标签，B*K
                ## From replicated labels get those of the top-K neighbours using the index yi (from top-k operation)

                retrieval_one_hot_train.resize_(batchSize * k_val, class_num).zero_()  # B*k,C
                retrieval_one_hot_train.scatter_(1, retrieval.view(-1, 1), 1)  #

                yd_transform = torch.exp(yd.clone().div_(cos_t))  ## Apply temperature to scores
                # yd_transform[...] = 1.0  ##To avoid using similarities only counts
                probs_corrected = torch.sum(
                    torch.mul(retrieval_one_hot_train.view(batchSize, -1, class_num),
                              yd_transform.view(batchSize, -1, 1)), 1)
                # 修正的prob指的是周围k个的onehot乘以修正后的相似度(F*F.t)

                probs_norm = probs_corrected / torch.sum(probs_corrected, dim=1)[:, None]  # B*C
                prob_temp = probs_norm[torch.arange(0, batchSize), labels[index]]

                discrepancy_measure2[index] = prob_temp
                agreement_measure2[index.data.cpu()] = (
                            torch.max(probs_norm, dim=1)[1] == batch_label).float().data.cpu()

    # 修正后的标签和原标签相似的做标记agreement_measure，说明该样本属于clean samples，属于集合T
    # discrepancy_measure2_pseudo_labels = \hat y
    if two_knn:
        discrepancy_measure = discrepancy_measure2
    else:
        discrepancy_measure = discrepancy_measure1
    # select examples
    # our sse setting
    agreement_measure = torch.zeros((len(labels),))  # N,1

    for i in range(class_num):
        idx_class = labels == i
        num_per_class = idx_class.sum()
        idx_class = (idx_class.float() == 1.0).float().nonzero().squeeze()

        discrepancy_class = discrepancy_measure[idx_class]  # 修正的距离分布
        k_corrected = sel_ratio * num_per_class
        if k_corrected > 0:
            top_clean_class_relative_idx = \
                torch.topk(discrepancy_class, k=int(k_corrected), largest=True, sorted=True)[1]
            #
            # 根据discrepancy_measure2，选出k_corrected个最小的index
            i_sel_index = idx_class[top_clean_class_relative_idx].long()
            # temp_record_selnum[i]=i_sel_index.size(0)

            agreement_measure[i_sel_index] = 1.0  # 对于误差前k小的样本agreement标记为1，N*1
            if corrected:
                '''
                we found the confidence is dense scattered, we should also select samples with the score
                that is same as the top selected
                '''
                the_val = discrepancy_class[top_clean_class_relative_idx[-1]]
                the_val_index = (discrepancy_class == the_val).float().nonzero().squeeze().long()
                agreement_measure[idx_class[the_val_index]] = 1.0
    # samples both should have same pseudo_labels all the time and high confidence.
    agreement_measure = agreement_measure * agreement_measure2
    # 根据相似度选出的同一语义标签的样本index

    with torch.no_grad():
        index_selected = torch.nonzero(agreement_measure, as_tuple=True)[0].cpu()  # N
        total_selected_num = len(index_selected)
        trainNoisyLabels = trainNoisyLabels.cpu().unsqueeze(1)  # N*1
        # total_num = len(trainNoisyLabels)
        noisy_pairs = torch.eq(trainNoisyLabels, trainNoisyLabels.t())  # N*N
        final_selected_pairs = torch.zeros_like(noisy_pairs).type(torch.uint8)

        # #暂存选择的同类clean samples
        selected_pairs = noisy_pairs[
            index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num), index_selected.unsqueeze(
                0).expand(total_selected_num, total_selected_num)].clone()
        final_selected_pairs[
            index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num), index_selected.unsqueeze(
                0).expand(total_selected_num, total_selected_num)] = selected_pairs.type(torch.uint8)
        final_selected_pairs = final_selected_pairs.type(torch.bool)

    if plot:
        if two_knn:
            plt.hist(discrepancy_measure2.cpu().numpy(), color='blue')
        else:
            plt.hist(discrepancy_measure1.cpu().numpy(), color='blue')
        plt.xlabel('score')
        plt.ylabel('freq')
        plt.title('freq/score')
        plt.savefig(f'./res/confidence/sse_confidence_epoch{epoch}.jpg')

    return agreement_measure.cuda(), final_selected_pairs.contiguous()


def pair_selection_by_selected(k_val, testloader, labels, class_num, cos_t, trainFeatures, set_onehot,
                               sel_ratio=0, two_knn=True, epoch=0):
    # the seleted samples set index at the first epoch should be used here
    # the threshold is set according to the portion of sel_ratio*(num of selected) samples
    trainFeatures = trainFeatures
    similiar_graph_all = torch.zeros(len(testloader.dataset), len(testloader.dataset))

    trainNoisyLabels = labels.clone().cuda()
    train_new_labels = labels.clone().cuda()
    discrepancy_measure1 = torch.zeros((len(testloader.dataset),)).cuda()
    discrepancy_measure2 = torch.zeros((len(testloader.dataset),)).cuda()
    agreement_measure = torch.zeros((len(testloader.dataset),))

    # weighted k-nn correction,做了两次近邻标签修正
    print('starting knn1....')
    with torch.no_grad():
        retrieval_one_hot_train = torch.zeros(k_val, class_num).cuda()
        for batch_idx, (inputs, _, index) in enumerate(testloader):
            batch_size = inputs.size(0)
            features = trainFeatures[index]

            dist = torch.mm(features, trainFeatures.t())  # B*N
            similiar_graph_all[index] = dist.cpu().detach()
            dist[torch.arange(dist.size(0)), index] = -1  # self-contrast set to -1

            yd, yi = dist.topk(k_val, dim=1, largest=True, sorted=True)  # yi:B*K
            candidates = labels.view(1, -1).expand(batch_size, -1)  # replicate the labels #B*N
            retrieval = torch.gather(candidates, 1, yi)  # gather获取指定index的元素；获得topk的label
            # size(B*K);yi 是index; from replicated labels to get those of the topk neighbors using the index yi

            retrieval_one_hot_train.resize_(batch_size * k_val, class_num).zero_()  # B*k,C
            # generate the k*batchsize one-hot encoding from neighboring labels
            # set of neighbouring labels is turned into one-hot encoding
            retrieval_one_hot_train.scatter_(1, retrieval.view(-1, 1), 1)  # scatter改变自身指定位置元素，变为one-hot向量

            yd_transform = torch.exp(yd.clone().div_(cos_t))  # apply temperature to score
            # yd_transform[...] = 1.0  # to avoid using similarities

            # 通过k近邻改变当前样本的
            probs_corrected = torch.sum(torch.mul(retrieval_one_hot_train.view(batch_size, -1, class_num),
                                                  yd_transform.view(batch_size, -1, 1)), 1)
            # (B,K,C)*(B,K,1).sum=(B*C)，按照yd的程度加权近邻k的独热编码
            probs_norm = probs_corrected / torch.sum(probs_corrected, dim=1)[:, None]

            # 选出每个样本最高的类别相似度然后对过低or过高修正
            prob_temp = probs_norm[torch.arange(0, batch_size), labels[index]]
            prob_temp[prob_temp <= 1e-2] = 1e-3
            prob_temp[prob_temp > (1 - 1e-2)] = 1 - 1e-3
            discrepancy_measure1[index] = -torch.log(prob_temp)

            # 根据修正后的重新分类标签
            sorted_pro, predictions_corrected = probs_norm.sort(1, True)
            new_labels = predictions_corrected[:, 0]

            train_new_labels[index] = new_labels
    # 修正后的标签和原标签相似的做标记agreement_measure，说明该样本属于clean samples，属于集合T

    if two_knn:
        print(f'starting knn2...')
        train_new_labels2 = train_new_labels.clone()  # N*1
        with torch.no_grad():
            retrieval_one_hot_train = torch.zeros(k_val, class_num).cuda()  #
            for batch_idx, (inputs, _, index) in enumerate(testloader):
                batchSize = inputs.size(0)
                features = trainFeatures[index]  # 64,256

                dist = torch.mm(features, trainFeatures.t())
                dist[torch.arange(dist.size()[0]), index] = -1  # Self-contrast set to -1

                yd, yi = dist.topk(k_val, dim=1, largest=True,
                                   sorted=True)  # Top-K similar scores and corresponding indexes .yi:B*K
                candidates = train_new_labels2.view(1, -1).expand(batchSize, -1)  # !!!这里变了 B*N
                retrieval = torch.gather(candidates, 1, yi)  # 对topk的近邻样本拿出来伪标签，B*K
                # From replicated labels get those of the top-K neighbours using the index yi (from top-k operation)

                retrieval_one_hot_train.resize_(batchSize * k_val, class_num).zero_()  # B*k,C
                retrieval_one_hot_train.scatter_(1, retrieval.view(-1, 1), 1)  #

                yd_transform = torch.exp(yd.clone().div_(cos_t))  # Apply temperature to scores
                # yd_transform[...] = 1.0  # To avoid using similarities only counts
                probs_corrected = torch.sum(
                    torch.mul(retrieval_one_hot_train.view(batchSize, -1, class_num),
                              yd_transform.view(batchSize, -1, 1)), 1)
                # 修正的prob指的是周围k个的onehot乘以修正后的相似度(F*F.t)

                probs_norm = probs_corrected / torch.sum(probs_corrected, dim=1)[:, None]  # B*C
                prob_temp = probs_norm[torch.arange(0, batchSize), labels[index]]
                prob_temp[prob_temp <= 1e-2] = 1e-3
                prob_temp[prob_temp > (1 - 1e-2)] = 1 - 1e-3

                discrepancy_measure2[index] = -torch.log(prob_temp)

    # 修正后的标签和原标签相似的做标记agreement_measure，说明该样本属于clean samples，属于集合T
    if two_knn:
        discrepancy_measure = discrepancy_measure2
    else:
        discrepancy_measure = discrepancy_measure1

    plt.hist(discrepancy_measure.clone().cpu().numpy(), bins=50, color="blue")
    '''
    data:必选参数，绘图数据
    bins:直方图的长条形数目，可选项，默认为10
    normed:是否将得到的直方图向量归一化，可选项，默认为0，代表不归一化，显示频数。normed=1，表示归一化，显示频率。
    facecolor:长条形的颜色
    edgecolor:长条形边框的颜色
    alpha:透明度
    '''
    plt.xlabel("scale")
    # 显示纵轴标签
    plt.ylabel("freq")
    # 显示图标题
    plt.title("freq/scale")
    plt.savefig(f'./res/confidence/freq_score_epoch{epoch}.jpg')

    # select examples
    set_index = torch.nonzero(set_onehot).squeeze()
    selected_confidence = discrepancy_measure[set_index]
    threshold = torch.quantile(selected_confidence, sel_ratio)
    print(f'newly set threshold: {threshold}')
    agreement_measure[discrepancy_measure < threshold] = 1.0
    print(f'discrepancy measure slice:{discrepancy_measure[:10]}')
    print(f'agreement_measure num:{agreement_measure.sum().item()}')

    with torch.no_grad():
        index_selected = torch.nonzero(agreement_measure, as_tuple=True)[0].cpu()  # N
        total_selected_num = len(index_selected)
        trainNoisyLabels = trainNoisyLabels.cpu().unsqueeze(1)  # N*1
        noisy_pairs = torch.eq(trainNoisyLabels, trainNoisyLabels.t())  # N*N
        final_selected_pairs = torch.zeros_like(noisy_pairs).type(torch.uint8)

        # #暂存选择的同类clean samples
        selected_pairs = noisy_pairs[
            index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num), index_selected.unsqueeze(
                0).expand(total_selected_num, total_selected_num)].clone()
        final_selected_pairs[
            index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num), index_selected.unsqueeze(
                0).expand(total_selected_num, total_selected_num)] = selected_pairs.type(torch.uint8)
        final_selected_pairs = final_selected_pairs.type(torch.bool)

    return agreement_measure.cuda(), final_selected_pairs.contiguous()



def pair_selection_cluster_correction(k_val, testloader, labels, class_num, trainFeatures,
                                        sel_ratio=0, plot=False, epoch=-1):
    # the confidence come from the pseudo-labels of the neighborhood.
    # only use pseudo-label as vote method
    trainNoisyLabels = labels.clone().cuda()
    # train_new_labels = labels.clone().cuda()
    discrepancy_measure = torch.zeros((len(testloader.dataset),)).cuda()
    similiar_graph_all = torch.zeros(len(testloader.dataset), len(testloader.dataset))

    print('starting knn1....')
    with torch.no_grad():
        retrieval_one_hot_train = torch.zeros(k_val, class_num).cuda()
        for batch_idx, (_, _, index) in enumerate(testloader):
            batch_size = index.size(0)
            features = trainFeatures[index]
            dist = torch.mm(features, trainFeatures.t())  # B*N

            similiar_graph_all[index] = dist.cpu().detach()
            dist[torch.arange(dist.size(0)), index] = -1  # self-contrast set to -1

            yd, yi = dist.topk(k_val, dim=1, largest=True, sorted=True)  # yi:B*k, yd:B*K
            candidates = labels.view(1, -1).expand(batch_size, -1)  # replicate the labels #B*N
            retrieval = torch.gather(candidates, 1, yi)  # gather获取指定index的元素；获得topk的label
            # size(B*K);yi 是index; from replicated labels to get those of the topk neighbors using the index yi

            retrieval_one_hot_train.resize_(batch_size * k_val, class_num).zero_()  # B*k,C
            # generate the k*batchsize one-hot encoding from neighboring labels
            # set of neighbouring labels is turned into one-hot encoding
            retrieval_one_hot_train.scatter_(1, retrieval.view(-1, 1), 1)  # scatter改变自身指定位置元素，变为one-hot向量
            yd_transform = torch.ones_like(yd).cuda()

            # 通过k近邻改变当前样本的, mul对应位相乘
            probs_corrected = torch.sum(torch.mul(retrieval_one_hot_train.view(batch_size, -1, class_num),
                                                  yd_transform.view(batch_size, -1, 1)), 1)  # BKC * BK1 .sum() = B*C
            # (B,K,C)*(B,K,1).sum=(B*C)，按照yd的程度加权近邻k的独热编码
            probs_norm = probs_corrected / torch.sum(probs_corrected, dim=1)[:, None]

            # 选出每个样本最高的类别相似度然后对过低or过高修正
            prob_temp = probs_norm[torch.arange(0, batch_size), labels[index]]
            prob_temp[prob_temp <= 1e-4] = 1e-4
            prob_temp[prob_temp > (1 - 1e-4)] = 1 - 1e-4
            discrepancy_measure[index] = -torch.log(prob_temp)

            # sorted_pro, predictions_corrected = probs_norm.sort(1, True)
            # new_labels = predictions_corrected[:, 0]

            # train_new_labels[index] = new_labels
            # agreement_measure[index.data.cpu()] = (torch.max(probs_norm, dim=1)[1] == labels[index]).float().data.cpu()
        # selected_examples = agreement_measure
    # 修正后的标签和原标签相似的做标记agreement_measure，说明该样本属于clean samples，属于集合T
    # our sse setting
    agreement_measure = torch.zeros((len(labels),)).cuda()  # N,1

    for i in range(class_num):
        idx_class = labels == i
        num_per_class = idx_class.sum()
        idx_class = (idx_class.float() == 1.0).float().nonzero().squeeze()
        discrepancy_class = discrepancy_measure[idx_class]  # 修正的距离分布

        k_corrected = sel_ratio * num_per_class
        if k_corrected >= 1:
            top_clean_class_relative_idx = \
                torch.topk(discrepancy_class, k=int(k_corrected), largest=False, sorted=True)[1]

            # 根据discrepancy_measure2，选出k_corrected个最小的index
            i_sel_index = idx_class[top_clean_class_relative_idx]
            agreement_measure[i_sel_index] = 1.0  # 对于误差前k小的样本agreement标记为1，N*1
        selected_examples = agreement_measure


    with torch.no_grad():
        index_selected = torch.nonzero(selected_examples, as_tuple=True)[0].cpu()  # N
        total_selected_num = len(index_selected)
        trainNoisyLabels = trainNoisyLabels.cpu().unsqueeze(1)  # N*1

        noisy_pairs = torch.eq(trainNoisyLabels, trainNoisyLabels.t())  # N*N if i,j have same label, [i,j]=1
        final_selected_pairs = torch.zeros_like(noisy_pairs).type(torch.uint8)

        # selected
        selected_pairs = noisy_pairs[
            index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num), index_selected.unsqueeze(
                0).expand(total_selected_num, total_selected_num)].clone()

        final_selected_pairs[
            index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num), index_selected.unsqueeze(
                0).expand(total_selected_num, total_selected_num)] = selected_pairs.type(torch.uint8)
        final_selected_pairs = final_selected_pairs.type(torch.bool)
        # finally pairs that are selected and have same labels are set to true
    if plot:
        plt.hist(discrepancy_measure.cpu().numpy(), color='blue')
        plt.xlabel('score')
        plt.ylabel('freq')
        plt.title('freq/score')
        plt.savefig(f'./res/confidence/sse_confidence_epoch{epoch}.jpg')

    return selected_examples.cuda(), final_selected_pairs.contiguous()

def pseudo_iterative(k_val, testloader, labels, class_num, cos_t, knn_times, trainFeatures, balance_class=True,
                   sel_ratio=0, plot=False, epoch=-1, corrected=False):
    '''
    简化版本
    kval: neighbors number of knn
    label: the mem_label, pseudo-labels obtained by feature prototypes
    '''

    similiar_graph_all = torch.zeros(len(testloader.dataset), len(testloader.dataset))

    trainNoisyLabels = labels.clone().cuda()
    train_labels = labels.clone().cuda()
    discrepancy_measure = torch.zeros((len(testloader.dataset),)).cuda()
    agreement_measure = torch.zeros((len(testloader.dataset),))

    # weighted k-nn correction,做了两次近邻标签修正

    with torch.no_grad():
        for i in range(knn_times):
            print(f'starting the {i+1}st knn....')
            retrieval_one_hot_train = torch.zeros(k_val, class_num).cuda()
            train_new_labels = train_labels
            for batch_idx, (_, _, index) in enumerate(testloader):
                batch_size = index.size(0)
                features = trainFeatures[index]

                # 当前batch的feature和其他所有样本的feature的cosine相似度距离
                dist = torch.mm(features, trainFeatures.t())  # B*N
                similiar_graph_all[index] = dist.cpu().detach()
                dist[torch.arange(dist.size(0)), index] = -1  # self-contrast set to -1

                # every sample have k-nearest neighbors
                yd, yi = dist.topk(k_val, dim=1, largest=True, sorted=True)  # yi:B*k, yd:B*K
                candidates = train_new_labels.view(1, -1).expand(batch_size, -1)  # replicate the labels #B*N
                retrieval = torch.gather(candidates, 1, yi)  # gather获取指定index的元素；获得topk的label
                # size(B*K);yi 是index; from replicated labels to get those of the topk neighbors using the index yi

                retrieval_one_hot_train.resize_(batch_size * k_val, class_num).zero_()  # B*k,C
                # generate the k*batchsize one-hot encoding from neighboring labels
                # set of neighbouring labels is turned into one-hot encoding
                # scatter_(dim, index, src),
                # dim=1, input[i][index[i][j]]的位置数值变为src[i][j]
                # retrieval.view后变为Bkc*1,
                retrieval_one_hot_train.scatter_(1, retrieval.view(-1, 1), 1)


                yd_transform = torch.exp(yd.clone().div_(cos_t))  # apply temperature to score
                # yd_transform[...] = 1.0  # to avoid using similarities

                # 通过k近邻改变当前样本的置信度, mul对应位相乘
                probs_corrected = torch.sum(torch.mul(retrieval_one_hot_train.view(batch_size, -1, class_num),
                                                      yd_transform.view(batch_size, -1, 1)), 1)
                # (B,K,C)*(B,K,1).sum=(B*C)，按照yd的程度加权近邻k的独热编码
                # 将数值归到（0，1）
                probs_norm = probs_corrected / torch.sum(probs_corrected, dim=1)[:, None]

                # 选出每个样本对于最初始的pseudo labels的prob_norm
                prob_temp = probs_norm[torch.arange(0, batch_size), labels[index]]
                prob_temp[prob_temp <= 1e-4] = 1e-4
                prob_temp[prob_temp > (1 - 1e-4)] = 1 - 1e-4
                discrepancy_measure[index] = -torch.log(prob_temp)

                # 根据修正后的结果重新分配标签
                sorted_pro, predictions_corrected = probs_norm.sort(1, True)
                new_labels = predictions_corrected[:, 0]

                train_labels[index] = new_labels
                agreement_measure[index.data.cpu()] = (torch.max(probs_norm, dim=1)[1] == labels[index]).float().data.cpu()
            selected_examples = agreement_measure


    if balance_class:
        # balance_class: different from previous versions, here it means: whether to select the same gamma ratio from each class
        # our sse setting
        # top gamma is top gamma of each class
        agreement_measure = torch.zeros((len(labels),)).cuda()  # N,1

        for i in range(class_num):
            idx_class = labels == i
            num_per_class = idx_class.sum()
            idx_class = (idx_class.float() == 1.0).float().nonzero().squeeze()
            discrepancy_class = discrepancy_measure[idx_class]

            k_corrected = sel_ratio * num_per_class
            if k_corrected >= 1:
                top_clean_class_relative_idx = \
                    torch.topk(discrepancy_class, k=int(k_corrected), largest=False, sorted=True)[1]

                # 选出k_corrected个最小的index
                i_sel_index = idx_class[top_clean_class_relative_idx]
                # 对于误差前k小的样本agreement标记为1，N*1
                agreement_measure[i_sel_index] = 1.0
                if corrected:
                    '''
                    we found the confidence is dense scattered, we should also select samples with the score
                    that is same as the top selected
                    '''
                    the_val = discrepancy_class[top_clean_class_relative_idx[-1]]
                    the_val_index = (discrepancy_class == the_val).float().nonzero().squeeze().long()
                    agreement_measure[idx_class[the_val_index]] = 1.0

        selected_examples = agreement_measure
        # selected_examples is the variable to be consistent with the other if s
        # 根据相似度选出的同一语义标签的样本index
        # print(f'temp_record_selnum:{temp_record_selnum}')

    with torch.no_grad():
        index_selected = torch.nonzero(selected_examples, as_tuple=True)[0].cpu()  # N
        total_selected_num = len(index_selected)
        trainNoisyLabels = trainNoisyLabels.cpu().unsqueeze(1)  # N*1

        noisy_pairs = torch.eq(trainNoisyLabels, trainNoisyLabels.t())  # N*N if i,j have same label, [i,j]=1
        final_selected_pairs = torch.zeros_like(noisy_pairs).type(torch.uint8)

        # selected
        selected_pairs = noisy_pairs[
            index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num), index_selected.unsqueeze(
                0).expand(total_selected_num, total_selected_num)].clone()
        final_selected_pairs[
            index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num), index_selected.unsqueeze(
                0).expand(total_selected_num, total_selected_num)] = selected_pairs.type(torch.uint8)
        final_selected_pairs = final_selected_pairs.type(torch.bool)
        # finally pairs that are selected and have same labels are set to true
    if plot:
        plt.hist(discrepancy_measure.cpu().numpy(), color='blue', )
        plt.xlabel('score')
        plt.ylabel('freq')
        plt.title('freq/score')
        plt.savefig(f'./res/confidence/sse_confidence_epoch{epoch}.jpg')

    return selected_examples.cuda(), final_selected_pairs.contiguous()
