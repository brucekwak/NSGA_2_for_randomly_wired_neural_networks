# [Reference #1] https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
# [Reference #2] https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py
# [Reference #3] https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151

import torch
import numpy as np
import random

import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from cutmix.cutmix import CutMix


def get_train_valid_loader_CIFAR10(data_path,
                           batch_size,
                           valid_size,
                           shuffle,
                           num_workers,
                           pin_memory):

    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # define transforms
    cutout_length = 16

    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Cutout(cutout_length),  # from nsga-net github
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        ])

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        ])
    
    # load the dataset
    train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    valid_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=valid_transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        random_seed = random.randint(1, 10000)
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return (train_loader, valid_loader)


def get_train_valid_loader_CIFAR10_cutmix(data_path,
                           batch_size,
                           valid_size,
                           shuffle,
                           num_workers,
                           pin_memory):

    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg


    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        ])

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        ])
    
    # load the dataset
    train_dataset_ = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    valid_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=valid_transform)

    #########################
    # CutMix 적용
    #########################
    train_dataset = CutMix(train_dataset_, num_class=10, beta=1.0, prob=0.5, num_mix=2)  # this is paper's original setting for cifar.

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        random_seed = random.randint(1, 10000)
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return (train_loader, valid_loader)


def get_test_loader_CIFAR10(data_path,
                           batch_size,
                           shuffle,
                           num_workers,
                           pin_memory):
    
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    
    
    normalize = transforms.Normalize(
        mean=CIFAR_MEAN,
        std=CIFAR_STD,
    )

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return data_loader



# [Reference] https://github.com/ianwhale/nsga-net/blob/master/misc/utils.py
class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img