from torch import nn


class DiceLoss(nn.Module):
    """
    DiceLoss is a PyTorch module that computes the Dice loss, which is commonly used 
    for evaluating the performance of image segmentation models.
    The Dice loss is defined as:
        Dice Loss = 1 - (2 * |X ∩ Y| + 1) / (|X| + |Y| + 1)
    where X is the predicted segmentation and Y is the ground truth segmentation.
    """
    def __init__(self):
        """
        Initializes the DiceLoss class.
        Calls the constructor of the parent class.
        """
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        """
        Compute the Dice loss between the inputs and targets.

        Args:
            inputs (torch.Tensor): Predicted tensor of shape (N, *) where * means any number of additional dimensions.
            targets (torch.Tensor): Ground truth tensor of shape (N, *) where * means any number of additional dimensions.
        Returns:
            torch.Tensor: Computed Dice loss.
        """
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss