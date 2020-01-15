# Learning-Loss-for-Active-Learning Pytorch Implementation

***
paper : https://arxiv.org/abs/1905.03677
author slide : https://www.slideshare.net/NaverEngineering/learning-loss-for-active-learning

All pictures are taken from the author's slide.

**It is not finished yet.**
***

Margin ranking loss

```python

class MarginRankingLoss_learning_loss(nn.Module):
    def __init__(self, margin=1.0):
        super(MarginRankingLoss_learning_loss, self).__init__()
        self.margin = margin
    def forward(self, inputs, targets):
        random = torch.randperm(inputs.size(0))
        pred_loss = inputs[random]
        pred_lossi = inputs[:inputs.size(0)//2]
        pred_lossj = inputs[inputs.size(0)//2:]
        target_loss = targets.reshape(inputs.size(0), 1)
        target_loss = target_loss[random]
        target_lossi = target_loss[:inputs.size(0)//2]
        target_lossj = target_loss[inputs.size(0)//2:]
        final_target = torch.sign(target_lossi - target_lossj)
        
        return F.margin_ranking_loss(pred_lossi, pred_lossj, final_target, margin=self.margin, reduction='mean')
        
```
