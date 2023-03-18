import os
import argparse
import math
import torch
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from data import *
from utils import *
from models import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, required = False)
    parser.add_argument('--batchsize', type=int, default=128, required = False)
    parser.add_argument('--epochs', type=int, default=200, required = False)
    parser.add_argument("--pretrained", type = str, default=None, required = False)
    parser.add_argument("--lr", type = float, default=1.5e-4, required = False)
    parser.add_argument("--momentum", type = float, default = 0.9, required = False)
    parser.add_argument("--weight_decay", type = float, default = 0.05, required = False)
    parser.add_argument("--dataset", type = str, required = False)
    parser.add_argument("--model", type = str, required = False)
    parser.add_argument("--local_rank", type = int, default = 0, required = False)
    parser.add_argument("--mask_ratio", type = float, default = 0.75, required = False)
    parser.add_argument('--warmup_epoch', type=int, default=200, required = False)

    parser.add_argument("--gpu_ids", type = str, required = True)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)
    train_dl, test_dl = get_dataloader(args)

    if args.pretrained is not None:
        model = MAE_ViT()
        model.load_state_dict(torch.load("checkpoint/model.pt"))
    
    model = ViT_Classifier(model.encoder, num_classes = 10).to(args.device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer, lr_sch = get_optim(args, model)
    acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

    for eph in range(args.epoch):
        model.train()
        losses = []
        acces = []
        for img, label in tqdm(iter(train_dl)):
            optimizer.zero_grad()
            img, label = img.to(args.device), label.to(args.device)
            logits = model(img)
            loss = loss_fn(logits, label)
            acc = acc_fn(logits, label)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            acces.append(acc.item())
        lr_sch.step()
        avg_train_loss = sum(losses) / len(losses)
        avg_train_acc = sum(acces) / len(acces)


        model.eval()
        with torch.no_grad():
            losses = []
            acces = []
            for img, label in tqdm(iter(test_dl)):
                img, label = img.to(args.device), label.to(args.device)
                logits = model(img)
                loss = loss_fn(logits, label)
                acc = acc_fn(logits, label)
                losses.append(loss.item())
                acces.append(acc.item())
            avg_val_loss = sum(losses) / len(losses)
            avg_val_acc = sum(acces) / len(acces)