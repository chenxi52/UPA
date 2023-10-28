import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import os.path
import cv2
import torchvision


def make_dataset(image_list, labels, append_root=None):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        else:
            if append_root:
                images = [(os.path.join(append_root, val.split()[0]), int(val.split()[1])) for val in image_list]
            else:
                images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB',append_root=None):
        imgs = make_dataset(image_list, labels,append_root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class ImageList_idx(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB',append_root=None):
        imgs = make_dataset(image_list, labels, append_root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)


class Pseudo_dataset(Dataset):
    def __init__(self, image_list, mem_labels, transform=None, mode='RGB', append_root=None):
        imgs = make_dataset(image_list, None, append_root=append_root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.imgs = imgs
        self.target = mem_labels
        self.transform = transform

        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, _ = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        target = self.target[index]

        return img, target, index

    def __len__(self):
        return len(self.imgs)


class ImageList_twice(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transform is not None:
            if type(self.transform).__name__ == 'list':
                img = [t(img) for t in self.transform]
            else:
                img = self.transform(img)

        return img, target, index

    def __len__(self):
        return len(self.imgs)


class Two_ImageList_idx(Dataset):
    def __init__(self, image_list, labels=None, transform1=None, transform2=None, target_transform=None, mode='RGB', append_root=None):
        imgs = make_dataset(image_list, labels, append_root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform1 = transform1
        self.transform2 = transform2
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)

        img1 = self.transform1(img)
        imgm1 = self.transform2(img)
        return img1, imgm1, target, index

    def __len__(self):
        return len(self.imgs)


class ImageList_multi_transform(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transform is not None:
            if type(self.transform).__name__ == 'list':
                img = [t(img) for t in self.transform]
            else:
                img = self.transform(img)

        return img, target, index

    def __len__(self):
        return len(self.imgs)


class Sel_dataset(Dataset):
    def __init__(self, sel_index, dataset):
        self.dataset = dataset
        sel_index_array = sel_index.int().cpu().clone().numpy()
        self.imgs = np.array(self.dataset.imgs)[sel_index_array]
        self.loader = self.dataset.loader
        self.transform = self.dataset.transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target, index

    def __len__(self):
        return len(self.imgs)
