import torch

def trainer(args, model, train_dl, optimizer, lr_sch, logger, eph):
    model.train()
    losses = []
    for img, label in train_dl:
        img, label = img.to(args.device), label.to(args.device)
        predicted_img, mask = model(img)
        optimizer.zero_grad()

        loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    lr_sch.step()
    avg_loss = sum(losses) / len(losses)
    # logger.write(f"[{eph + 1} / {args.epochs}]   avg_loss : {avg_loss}")
    return avg_loss