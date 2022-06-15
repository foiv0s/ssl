import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def tanh_clip(x, clip_val=10.):
    '''
    soft clip values to the range [-clip_val, +clip_val]
    '''
    if clip_val is not None:
        x_clip = clip_val * torch.tanh((1. / clip_val) * x)
    else:
        x_clip = x
    return x_clip


def loss_xent(logits, labels, ignore_index=-1, weights=None, clip=10):
    '''
    compute multinomial cross-entropy, for e.g. training a classifier.
    '''
    if weights is not None:
        xent = F.cross_entropy(tanh_clip(logits, clip), labels,
                               ignore_index=ignore_index, weight=weights)
    else:
        xent = F.cross_entropy(tanh_clip(logits, clip), labels,
                               ignore_index=ignore_index)
    lgt_reg = 1e-3 * (logits ** 2.).mean()
    return xent + lgt_reg
