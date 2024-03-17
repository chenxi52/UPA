from __future__ import print_function
import torch
import warnings
warnings.filterwarnings('ignore')


def pair_selection_v1(k_val, test_loader, labels, class_num, cos_t, knn_times, train_features, balance_class=True,
                   sel_ratio=0, corrected=False):
    '''
    k_val:  neighbors number of knn
    labels: pseudo-labels obtained from feature prototypes
    '''
    similarity_graph_all = torch.zeros(len(test_loader.dataset), len(test_loader.dataset))
    train_noisy_labels = labels.clone().cuda()
    train_labels = labels.clone().cuda()
    discrepancy_measure = torch.zeros((len(test_loader.dataset),)).cuda()
    agreement_measure = torch.zeros((len(test_loader.dataset),))

    with torch.no_grad():
        for i in range(knn_times):
            print(f'starting the {i+1}st knn....')
            retrieval_one_hot_train = torch.zeros(k_val, class_num).cuda()
            train_new_labels = train_labels.clone()
            for batch_idx, (_, _, index) in enumerate(test_loader):
                batch_size = index.size(0)
                features = train_features[index]

                # similarity graph
                dist = torch.mm(features, train_features.t())  
                similarity_graph_all[index] = dist.cpu().detach()
                dist[torch.arange(dist.size(0)), index] = -1  # self-contrast set to -1

                # sample k-nearest neighbors
                yd, yi = dist.topk(k_val, dim=1, largest=True, sorted=True)  
                candidates = train_new_labels.view(1, -1).expand(batch_size, -1) 
                retrieval = torch.gather(candidates, 1, yi)

                retrieval_one_hot_train.resize_(batch_size * k_val, class_num).zero_() 
                retrieval_one_hot_train.scatter_(1, retrieval.view(-1, 1), 1)

                yd_transform = torch.exp(yd.clone().div_(cos_t))  
                probs_corrected = torch.sum(torch.mul(retrieval_one_hot_train.view(batch_size, -1, class_num),
                                                      yd_transform.view(batch_size, -1, 1)), 1)
                probs_norm = probs_corrected / torch.sum(probs_corrected, dim=1)[:, None]

                prob_temp = probs_norm[torch.arange(0, batch_size), labels[index]]
                prob_temp[prob_temp <= 1e-4] = 1e-4
                prob_temp[prob_temp > (1 - 1e-4)] = 1 - 1e-4
                discrepancy_measure[index] = -torch.log(prob_temp)

                # update the labels
                sorted_pro, predictions_corrected = probs_norm.sort(1, True)
                new_labels = predictions_corrected[:, 0]
                train_labels[index] = new_labels
                agreement_measure[index.data.cpu()] = (torch.max(probs_norm, dim=1)[1] == labels[index]).float().data.cpu()
            selected_examples = agreement_measure


    if balance_class:
        # select the top k_corrected samples for each class
        agreement_measure = torch.zeros((len(labels),)).cuda()  
        for i in range(class_num):
            idx_class = labels == i
            num_per_class = idx_class.sum()
            idx_class = (idx_class.float() == 1.0).float().nonzero().squeeze()
            discrepancy_class = discrepancy_measure[idx_class]

            k_corrected = sel_ratio * num_per_class
            if k_corrected >= 1:
                top_clean_class_relative_idx = \
                    torch.topk(discrepancy_class, k=int(k_corrected), largest=False, sorted=True)[1]

                i_sel_index = idx_class[top_clean_class_relative_idx]
                agreement_measure[i_sel_index] = 1.0
                if corrected:
                    the_val = discrepancy_class[top_clean_class_relative_idx[-1]]
                    the_val_index = (discrepancy_class == the_val).float().nonzero().squeeze().long()
                    agreement_measure[idx_class[the_val_index]] = 1.0
        selected_examples = agreement_measure

    with torch.no_grad():
        index_selected = torch.nonzero(selected_examples, as_tuple=True)[0].cpu()  
        total_selected_num = len(index_selected)
        train_noisy_labels = train_noisy_labels.cpu().unsqueeze(1)  

        noisy_pairs = torch.eq(train_noisy_labels, train_noisy_labels.t()) 
        final_selected_pairs = torch.zeros_like(noisy_pairs).type(torch.uint8)

        selected_pairs = noisy_pairs[
            index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num), index_selected.unsqueeze(
                0).expand(total_selected_num, total_selected_num)].clone()
        final_selected_pairs[
            index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num), index_selected.unsqueeze(
                0).expand(total_selected_num, total_selected_num)] = selected_pairs.type(torch.uint8)
        final_selected_pairs = final_selected_pairs.type(torch.bool)
    return selected_examples.cuda(), final_selected_pairs.contiguous()
