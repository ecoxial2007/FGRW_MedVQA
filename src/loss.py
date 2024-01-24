import torch
from torch import nn
import torch.nn.functional as F

## eps label smooth
def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self,  reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, preds, target, alpha=0):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = -log_preds.sum(dim=-1)
        nll = F.nll_loss(log_preds, target, reduction='none')
        return reduce_loss(alpha * loss/n + (1 - alpha) * nll, self.reduction)
