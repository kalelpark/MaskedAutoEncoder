from __future__ import print_function
import os
import argparse
import time

from data import *
from models import *
from utils import *
from runner.run import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type = int, default = 0, required = False)         
    parser.add_argument("--imgsize", type = int, default = 512, required = False)
    parser.add_argument("--crop", type = int, default = 512, required = False)
    parser.add_argument("--epochs", type = int, default = 2000, required = False)
    parser.add_argument("--batchsize", type = int, default = 512, required = False)
    parser.add_argument("--lr", type = float, default=1.5e-4, required = False)
    parser.add_argument("--momentum", type = float, default = 0.9, required = False)
    parser.add_argument("--weight_decay", type = float, default = 0.05, required = False)
    parser.add_argument("--dataset", type = str, required = False)
    parser.add_argument("--model", type = str, required = False)
    parser.add_argument("--local_rank", type = int, default = 0, required = False)
    parser.add_argument("--mask_ratio", type = float, default = 0.75, required = False)
    parser.add_argument('--warmup_epoch', type=int, default=200, required = False)
    parser.add_argument("--ckpt", type = str, default = "checkpoint/model_1.pt", required = False)

    parser.add_argument("--gpu_ids", type = str, required = True)
    
    args = parser.parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)
    logger = get_logger(args)

    model = MAE_ViT().to(args.device)
    train_dl, test_dl = get_dataloader(args)
    optimizer, lr_sch = get_optim(args, model)
    logger.write("[Get] model, dataset, optimizer, lr_sch")
    
    runner(args, model, train_dl, test_dl, optimizer, lr_sch, logger)
    logger.write("[End]") 