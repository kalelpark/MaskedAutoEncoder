import math
import torch
import torch.optim as optim

def get_optim(args, model):
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr * args.batchsize / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.epochs * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    return optim, lr_scheduler