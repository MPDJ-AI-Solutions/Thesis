import torchvision.models
import torch
import torch.nn as nn


class Backbone(nn.Module):
    """
    TODO: write docs
    """
    def __init__(self, input_size: int = 8, pretrained: bool = True):
        super(Backbone, self).__init__()
        self.input_size = input_size
        custom_resnet101 = torchvision.models.resnet101()

        # Change first layer for RGB + SWIR images input (input_size equals to considered wavelengths)
        custom_resnet101.conv1 = nn.Conv2d(
            input_size, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

        if pretrained:
            weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V1
            pretrained_weights = torchvision.models.resnet101(weights=weights).conv1.weight.data
            new_weights = torch.zeros(64, self.input_size, 7, 7)

            # Assign weights for RGB
            new_weights[:, :3, :, :] = pretrained_weights
            # Init weights for SWIR
            new_weights[:, 3:, :, :] = pretrained_weights.mean(dim=1, keepdim=True)

            # Save new weights
            custom_resnet101.conv1.weight.data = new_weights

        # Remove last 2 layers - use only feature extraction
        self.cnn_backbone = nn.Sequential(*list(custom_resnet101.children())[:-2])

    def forward(self, x):
        return self.cnn_backbone(x)
