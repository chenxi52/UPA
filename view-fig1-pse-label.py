# load source model
# select right pseudo-label samples
# visulize all& clean& noisy

import torch
import numpy as np 
import random
import argparse
from trainer.engine import Upa
from scipy.spatial.distance import cdist
from utils.tsne_utils import plot, MOUSE_10X_COLORS, plot_with_two_domains
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import os
SEED = 2021
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

parser = argparse.ArgumentParser(description='SHOT')
parser.add_argument('--dset', type=str, default='VISDA-C')
parser.add_argument('--output', type=str, default='san')
parser.add_argument('--output_dir_src', type=str, default='res/ckps/source/uda')
parser.add_argument('--net', type=str, default='resnet101',
                        help="alexnet, vgg16, resnet50, resnet101,vit")
parser.add_argument('--bottleneck', type=int, default=256)
parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn", "bn_drop"])
parser.add_argument('--aug', type=str, default='mocov2', help='strong augmentation type')
args = parser.parse_args()
args.out_file = open(os.path.join(args.output_dir_src, 'log_.txt'), 'w')
args.distance='cosine'
args.max_epoch = 1
args.output_dir_src = args.output_dir_src + '/' + args.dset + '/' + 'T'
print('model path', args.output_dir_src)
args.batch_size = 64
args.worker = 4
args.da = 'uda'
folder = 'datasets/'
# visda-c dataset
args.name = 'visda-c_test_source'
names = ['train', 'validation']
args.class_num = 12
args.t_dset_path = folder + args.dset + '/' + names[1] + '_list.txt'
args.test_dset_path = folder + args.dset + '/' + names[1] + '_list.txt'

# train the source model again
upaBuilder = Upa(args)
upaBuilder.encoder.eval()
upaBuilder.netC.eval()
test_dataset = upaBuilder.dsets['test']
upaBuilder.loader['test']=DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=args.worker,
                                      drop_last=False, sampler=None, pin_memory=True)
mem_label, all_fea, initc, all_label, all_output = upaBuilder.obtain_label(False)
print(f'len of dataset:{len(mem_label)}')
n_per_class = 100
n_class = 5
class_order = [0, 2, 4, 6, 8, 10]
# class_order = [0, 1, 2, 3, 4, 5]
n_samples = 2000
start = 2000
all_fea = all_fea.cpu().numpy()
all_label = all_label.cpu().numpy()
x = all_fea[start:start+n_samples]
y = all_label[start:start+n_samples]
mem_label = mem_label[start:start+n_samples]

class_mask = np.isin(y, class_order)
x = x[class_mask]
y= y[class_mask]
mem_label = mem_label[class_mask]

# 选出来 class_order 中的样本
true_pse_label = (mem_label == y)
print ('true_pse_label', np.sum(true_pse_label))
x_clean = x[true_pse_label]
y_clean = y[true_pse_label]
x_noise = x[~true_pse_label]
y_noise = y[~true_pse_label]
x_cat = np.concatenate([x_clean, x_noise],axis=0)
x_cat = x_cat.reshape((x_cat.shape[0], -1))
y_cat = np.concatenate([y_clean, y_noise], axis=0).astype(np.int)

n_clean = np.sum(true_pse_label)
y_cat = tuple(y_cat)
print ('start tsne...')
tsne = TSNE(n_components=2, random_state=0, perplexity=30, n_iter=1000)
transformed_data = tsne.fit_transform(x_cat)
print ('tsne done...')

# spltit
plot_with_two_domains(transformed_data,y_cat,colors=MOUSE_10X_COLORS, source_size=n_clean,img_dir='res/pic/re-source', draw_legend=False)
plot(transformed_data[:n_clean],y_clean,colors=MOUSE_10X_COLORS, alpha=1, s=40, img_dir='res/pic/re-source-clean', draw_legend=False, marker='*' )
plot(transformed_data[n_clean:],y_noise,colors=MOUSE_10X_COLORS, alpha=0.2, s=40, img_dir='res/pic/re-source-noisy', draw_legend=False,marker='o' )