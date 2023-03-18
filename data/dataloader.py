import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from .transform import *

def get_dataloader(args):
    train_trans, test_trans = get_transform()
    train_dataset = datasets.CIFAR10("data", train = True, download = True, transform = train_trans)
    test_dataset = datasets.CIFAR10("data", train = True, download = True, transform = test_trans)

    train_loader = DataLoader(train_dataset, batch_size = args.batchsize, shuffle = True, num_workers = 4)
    test_loader = DataLoader(test_dataset, batch_size = args.batchsize, shuffle = False, num_workers = 4)
    return train_loader, test_loader