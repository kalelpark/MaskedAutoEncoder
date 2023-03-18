import sys
import os
import random
import time

import numpy as np
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F












"""default Setting."""

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = False

class Logger(object):
    def __init__(self, local_rank = 0, no_save = False):
        self.terminal = sys.stdout
        self.file = None
        self.local_rank = local_rank
        self.no_save = no_save
    
    def open(self, fp, mode = None):
        if mode is None:
            mode = "a"
        if self.local_rank == 0:
            self.file = open(fp, mode)
    
    def write(self, msg, is_terminal = 1, is_file = 1):
        if msg[-1] != "\n":
            msg += "\n"
        
        if self.local_rank == 0:
            if "\r" in msg:
                is_file = 0
            
            if is_terminal == 1:
                self.terminal.write(msg)
                self.terminal.flush()
            if is_file == 1 and not self.no_save:
                self.file.write(msg)
                self.file.flush()

    def flush(self):
        pass

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_logger(args):
    temp = time.gmtime()
    tar = map(lambda x : str(x), list(temp)[1:-3])
    time_logger = ".".join(tar)
    logger = Logger(local_rank = args.local_rank)
    logger.open(f"log/{time_logger}.txt", mode = "w")
    logger.write("[start]")
    
    return logger
    

