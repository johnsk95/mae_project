# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
import pickle
import torch
import random
import numpy as np

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from torch.utils.data import Dataset

class PairedTransform:
    def __init__(self, size, scale=(0.2, 1.0)):
        # self.size = size
        self.size = (224, 224) # Hardcoded to 224x224 - John
        self.scale = scale
        
    def __call__(self, img, mask):
        # Get crop parameters
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            img, self.scale, (0.75, 1.333333333))
        
        # Apply same crop to both
        img = transforms.functional.resized_crop(
            img, i, j, h, w, self.size, interpolation=PIL.Image.BICUBIC)
        mask = transforms.functional.resized_crop(
            mask, i, j, h, w, self.size, interpolation=PIL.Image.NEAREST)
        
        # Apply same flip to both
        if random.random() > 0.5:
            img = transforms.functional.hflip(img)
            mask = transforms.functional.hflip(mask)
            
        return img, mask

# Custom Dataset for COCO-Stuff
class CocoStuffDataset(Dataset):
    def __init__(self, label_file, args, transform=None):
        with open(label_file, 'rb') as f:
            self.data = pickle.load(f)
        self.image_paths = list(self.data.keys())
        self.labels = list(self.data.values())
        self.args = args
        self.paired_transform = PairedTransform(self.args.input_size)
        self.image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        self.mask_transforms = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = '../ContextualBias/' + self.image_paths[idx]
        mask_path = '../ContextualBias/Data/cocostuff/annotations/all/' + self.image_paths[idx].split('/')[-1].split('_')[-1].replace('jpg', 'png')
        label, co_occur = self.labels[idx]

        image = PIL.Image.open(image_path).convert('RGB')
        mask = PIL.Image.open(mask_path).convert('L')
        
        # Apply paired transforms
        image, mask = self.paired_transform(image, mask)

        # Apply separate transforms
        image = self.image_transforms(image)
        # mask = self.mask_transforms(mask)
        mask = torch.tensor(np.array(mask).astype(np.int16)).unsqueeze(0)
        
        # return image, torch.tensor(label), torch.argmax(torch.tensor(label), dim=0)  # Convert one-hot to class index
        return (image, mask), torch.tensor(co_occur) # Convert one-hot to class index


def build_dataset(is_train, args):
    # transform = build_transform(is_train, args)
    # transform_mask = build_transform_mask(is_train, args)
    label_file = args.data_path + '/labels_train_cooccur.pkl' if is_train else args.data_path + '/labels_test.pkl'
    dataset = CocoStuffDataset(label_file=label_file, args=args)
    # dataset = CocoStuffDataset(label_file=label_file, transform=transform, transform_mask=transform_mask)
    return dataset



def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


