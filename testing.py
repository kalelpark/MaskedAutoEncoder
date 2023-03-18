import os
import argparse
import math
import torch
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm
from models import *

val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
# print(val_dataset[0][0].size())
val_img = torch.stack([val_dataset[i][0] for i in range(16)])
model = MAE_ViT()
pred_img, mask = model(val_img)
pred_img = pred_img * mask + pred_img * (1 - mask)

img = torch.cat([val_img * (1 - mask), pred_img, val_img], dim = 0)
print("temper : ", img.size())
img = rearrange(img, "(v h1 w1) c h w -> c (h1 h) (w1 v w)", w1 = 2, v = 3)
print("temper : ", img.size())
