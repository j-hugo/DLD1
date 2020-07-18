import torch.nn.functional as F
import torch
from torch import einsum
from functools import partial
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union

from torch import Tensor
def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def gdice_loss(pred, target, smooth = 1e-5):
    pc = pred
    tc = target
    w = 1 / ((einsum("bcwh->bc", tc).type(torch.float32) + 1e-10) ** 2) #shape Batch x Classes
    intersection = w * einsum("bcwh,bcwh->bc", pc, tc) #shape Batch x Classes
    union = w * (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc)) #shape Batch x Classes
    numerator = (einsum("bc->b", intersection) + 1e-10) #shape Batch x Classes
    denominator = (einsum("bc->b", union) + 1e-10) #shape Batch x Classes
    divided = 1 - 2 * (numerator / denominator) #Shape Batch
    loss = divided.mean() #scalar
    return loss

def dice_coeff(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    dice = ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth))
    return dice

def meta_dice(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8) -> float:
    label = label.type(torch.int)
    pred = pred.type(torch.int)
    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
    sum_sizes: Tensor = (einsum(sum_str, [label]) + einsum(sum_str, [pred])).type(torch.float32)
    dices: Tensor = (2 * inter_size + smooth) / (sum_sizes + smooth)
    return dices

def intersection(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])
    return a & b

def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())

dice_coef = partial(meta_dice, "bcwh->bc")

def calc_loss(pred, target, metrics, bce_weight=1.25, dc_weight=1.0):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * dc_weight

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    print("{}: {}".format(phase, ", ".join(outputs)))