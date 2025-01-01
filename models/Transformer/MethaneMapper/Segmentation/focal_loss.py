import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    This class implements the Focal Loss function, which is designed to address class imbalance by down-weighting 
    the loss assigned to well-classified examples. It is particularly useful for tasks with highly imbalanced classes.
    
    Args:
        alpha (float, optional): Balancing factor for the positive class. Default is 0.25.
        gamma (float, optional): Focusing parameter that reduces the relative loss for well-classified examples, 
                                 putting more focus on hard, misclassified examples. Default is 2.0.
        reduction (str, optional): Specifies the reduction to apply to the output: 'mean' (default), 'sum', or 'none'.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Initialize the FocalLoss class.

        Args:
            alpha (float, optional): Balancing factor for positive/negative examples. Default is 0.25.
            gamma (float, optional): Focusing parameter to adjust the rate at which easy examples are down-weighted. Default is 2.0.
            reduction (str, optional): Specifies the reduction to apply to the output. Options are 'mean', 'sum', or 'none'. Default is 'mean'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Balancing factor
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction  # How to reduce the loss ('mean', 'sum', or 'none')

    def forward(self, inputs, targets):
        """
        Forward pass for the focal loss computation.
        
        Args:
            inputs (torch.Tensor): The input predictions (logits) from the model.
            targets (torch.Tensor): The ground truth binary labels.
        Returns:
            torch.Tensor: The computed focal loss. The reduction type (mean, sum, or none) 
                          determines the final shape of the returned tensor.
        Raises:
            ValueError: If an unknown reduction type is specified.
        """
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
