import torch
import torch.nn as nn


def dice_loss_logits(logits, target, smooth=1.0):
    probs = torch.sigmoid(logits)
    probs = probs.view(-1)
    target = target.view(-1)
    intersection = (probs * target).sum()
    return 1 - ((2. * intersection + smooth) /
                (probs.sum() + target.sum() + smooth))


def dice_coeff(pred, target, smooth=1.0):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred, target, smooth=1.0):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def get_losses(device):
    pos_weight = torch.tensor([50.0]).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def combined(logits, target):
        return bce(logits, target) + dice_loss_logits(logits, target)

    return bce, combined
