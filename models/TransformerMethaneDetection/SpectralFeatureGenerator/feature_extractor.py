import torch
import torch.nn as nn
import torchvision

class FeatureExtractor(nn.Module):
    """
    Feature extractor for single-channel filtered images using ResNet-50.
    """
    def __init__(self, d_model: int = 256):
        super(FeatureExtractor, self).__init__()
        self.resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Adjust the first conv layer to accept 1 channel
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False
        )

        for name, param in self.resnet.named_parameters():
            if name not in ["conv1.weight", "conv1.bias"]:
                param.requires_grad = False
                
        # Project output to d_model channels
        self.project = nn.Conv2d(in_channels=2048, out_channels=d_model, kernel_size=1)
        self.cnn_backbone = nn.Sequential(*list(self.resnet.children())[:-2])


    def forward(self, filtered_image: torch.Tensor) -> torch.Tensor:
        cnn_features = self.cnn_backbone(filtered_image)
        return self.project(cnn_features)
