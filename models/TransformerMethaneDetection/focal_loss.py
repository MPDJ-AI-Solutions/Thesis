import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Balancing factor
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction  # How to reduce the loss ('mean', 'sum', or 'none')

    def forward(self, inputs, targets):
        # Apply sigmoid to the inputs for binary classification
        inputs = torch.sigmoid(inputs)

        # Compute cross-entropy loss (binary cross-entropy for binary classification)
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        # Calculate the focal weight
        p_t = targets * inputs + (1 - targets) * (1 - inputs)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma

        # Apply focal loss
        loss = focal_weight * BCE_loss

        # Reduce the loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Unknown reduction type: {self.reduction}")
