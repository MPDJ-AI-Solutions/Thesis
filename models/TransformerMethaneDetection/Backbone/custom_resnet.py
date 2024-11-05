import torch
import torch.nn as nn
import torchvision.models as models

class CustomResnet(nn.Module):
    """
    Custom resnet-50 used for image feature extraction before encoding.
    """
    def __init__(self, in_channels: int = 3, out_channels:int = 2048):
        super(CustomResnet, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        
        for name, param in self.resnet.named_parameters():
            if name not in ["conv1.weight", "conv1.bias"]:
                param.requires_grad = False
                
        self.cnn = nn.Sequential(*list(self.resnet.children())[:-2])
        self.project_layer = nn.Conv2d(in_channels=2048, out_channels=out_channels, kernel_size=1)


    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.cnn(image)
        return self.project_layer(features)
