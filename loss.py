import torch.nn.functional as F
import torch

# adpated from https://github.com/usuyama/pytorch-unet
def dice_coef(pred, target, smooth = 1.):
    """
        Calculate the dice coefficient

        Args:
            pred: predictions from the model 
            target: ground truths
            smooth: to prevent division by zero. 
                    Also, having a larger smooth value (also known as Laplace smooth, or Additive smooth) can be used to avoid overfitting. 
                    The larger the smooth value the closer the following term is to 1 (if everything else is fixed).

        Return:
            dice.mean(): the average value of dice coefficients  
    """
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    dice = ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth))
    return dice.mean()

# adapted from https://github.com/usuyama/pytorch-unet
def calc_loss(pred, target, metrics, bce_weight=1.0, dc_weight=1.0):
    """
        Calculate the loss

        Args:
            pred: predictions from the model 
            target: ground truths
            metrics: a dictionary to save losses
            bce_weight: a weight for binary cross entropy
            dc_weight: a weight for dice loss 

        Return:
            loss: the combination of dice loss and binary cross entropy
    """
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)
    dice_loss = 1 - dice_coef(pred, target)
    loss = bce * bce_weight + dice_loss * dc_weight

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice_loss'] += dice_loss.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

# adapted from https://github.com/usuyama/pytorch-unet
def print_metrics(metrics, epoch_samples, phase):
    """
        Print out metrics

        Args:
            metrics: a dictionary which has calculated losses
            epoch_samples: the total number of samples for one epoch 
            phase: a phase of the training

        Return:
            loss: the combination of dice loss and binary cross entropy
    """
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    print("{}: {}".format(phase, ", ".join(outputs)))