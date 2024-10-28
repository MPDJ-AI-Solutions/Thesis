import torchvision.models
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """
    TODO: write docs
    """
    def __init__(self, input_size: int = 8, pretrained: bool = True):
        super(FeatureExtractor, self).__init__()
        self.input_size = input_size
        resnet = torchvision.models.resnet50()

    def forward(self, x):
        ...
