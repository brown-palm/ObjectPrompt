import torch
from torch import nn


class LTAWeightedLoss(nn.Module):
    def __init__(self, loss_wts_heads, loss_wts_temporal):
        super(LTAWeightedLoss, self).__init__()
        # self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.loss_fun = nn.CrossEntropyLoss(reduction='mean')
        self.loss_wts_heads = loss_wts_heads
        self.loss_wts_temporal = loss_wts_temporal

    def forward(self, logits, targets, mask=None):
        '''
        logits: [(B, Z, #verbs), (B, Z, #nouns)]
        targets: (B, Z, 2)
        mask: (B, Z) or None
        '''

        loss_wts = self.loss_wts_heads
        losses = [0, 0]
        for head_idx in range(len(logits)):
            pred_head = logits[head_idx]  # (B, Z, C)
            for seq_idx in range(pred_head.shape[1]):
                losses[head_idx] += loss_wts[head_idx] * self.loss_fun(
                    pred_head[:, seq_idx], targets[:, seq_idx, head_idx]
                )
        return sum(losses), None