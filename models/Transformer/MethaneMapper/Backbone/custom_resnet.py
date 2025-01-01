import torch
import torch.nn as nn
import torchvision.models as models

class CustomResnet(nn.Module):
    """
    Custom resnet-50 used for image feature extraction before encoding.
    """
    def __init__(self, in_channels: int = 3, out_channels:int = 2048):
        """
        Initializes the CustomResnet model.
        Args:
            in_channels (int): Number of input channels. Default is 3.
            out_channels (int): Number of output channels. Default is 2048.
        Attributes:
            resnet (torchvision.models.ResNet): Pretrained ResNet-50 model with modified first convolutional layer.
            cnn (torch.nn.Sequential): Sequential model containing all layers of ResNet-50 except the last two.
            project_layer (torch.nn.Conv2d): Convolutional layer to project the output to the desired number of channels.
        """
        super(CustomResnet, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.resnet.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        
        for name, param in self.resnet.named_parameters():
            if name not in ["conv1.weight", "conv1.bias"]:
                param.requires_grad = False
            else:
                param.requires_grad = True

        self.cnn = nn.Sequential(*list(self.resnet.children())[:-2])
        self.project_layer = nn.Conv2d(in_channels=2048, out_channels=out_channels, kernel_size=1)


    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the custom ResNet model.

        Args:
            image (torch.Tensor): Input image tensor of shape (N, C, H, W) where
                                  N is the batch size, C is the number of channels,
                                  H is the height, and W is the width.

        Returns:
            torch.Tensor: Output tensor after passing through the CNN and projection layer.
        """
        features = self.cnn(image)
        return self.project_layer(features)
