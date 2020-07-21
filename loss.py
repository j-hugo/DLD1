import torch.nn.functional as F
import torch


def dice_coef(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    dice = ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth))
    return dice.mean()


def calc_loss(pred, target, metrics, bce_weight=1.0, dc_weight=1.0):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)
    dice_loss = 1 - dice_coef(pred, target)
    loss = bce * bce_weight + dice_loss * dc_weight

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice_loss'] += dice_loss.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    print("{}: {}".format(phase, ", ".join(outputs)))