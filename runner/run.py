from .train import *
from .infer import *
import wandb

def runner(args, model, train_dl, test_dl, optimizer, lr_sch, logger):
    logger.write("[START] Runner")
    total_loss = 1e10
    for eph in range(args.epochs):
        avg_loss = trainer(args, model, train_dl, optimizer, lr_sch, logger, eph)
        logger.write(f"[{str(eph + 1)} / {str(args.epochs)}] : Avg_Loss : {str(avg_loss)}")

        if total_loss > avg_loss:
            total_loss = avg_loss
            torch.save(model.state_dict(), args.ckpt)
            logger.write("[SAVE] model")