from torch import nn
from torchvision.models import vit_b_16
from torchvision.transforms import transforms


class CustomViT(nn.Module):
    def __init__(self, num_channels=9, num_classes=2):
        super(CustomViT, self).__init__()
        # Load pre-trained ViT model
        self.vit = vit_b_16(weights=None)  # Use pretrained weights if desired

        # Modify the input embedding layer to accept `num_channels`
        self.vit.conv_proj = nn.Conv2d(num_channels, self.vit.conv_proj.out_channels,
                                       kernel_size=self.vit.conv_proj.kernel_size,
                                       stride=self.vit.conv_proj.stride,
                                       padding=self.vit.conv_proj.padding,
                                       bias=(self.vit.conv_proj.bias is not None))

        # Modify the classifier head for binary classification
        self.vit.heads = nn.Sequential(
            nn.Linear(self.vit.heads.head.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        self.resize = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        self.normalize = transforms.Normalize(mean=[0.5] * num_channels, std=[0.5] * num_channels)

    def forward(self, x):
        x = self.resize(x)
        x = self.normalize(x)
        return self.vit(x)