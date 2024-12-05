import torch
import torch.nn as nn


class MethaNetClassifier(nn.Module):
    def __init__(self, image_h: int, image_w: int, in_channels:int = 8, num_classes:int = 2):
        super(MethaNetClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            # First Conv Layer: 6 filters of size (2,2) & ReLU
            nn.Conv2d(in_channels, 6, kernel_size=2, stride=1, padding=0),  # Input channels = 8
            nn.ReLU(),

            # Second Conv Layer: 12 filters of size (3,3) & ReLU
            nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),

            # Max Pooling of size (2,2)
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Third Conv Layer: 16 filters of size (4,4) & ReLU
            nn.Conv2d(12, 16, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),

            # Max Pooling of size (2,2)
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Dropout with p = 0.2
            nn.Dropout(p=0.2),

            # Fourth Conv Layer: 16 filters of size (4,4) & ReLU
            nn.Conv2d(16, 16, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),

            # Max Pooling of size (2,2)
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            # Fully Connected Layer: 64 neurons & ReLU
            nn.Flatten(),
            nn.Linear(16 * 61 * 61, 64),  # Adjust dimensions for 512x512 input
            nn.ReLU(),

            # Fully Connected Layer: 32 neurons & ReLU
            nn.Linear(64, 32),
            nn.ReLU(),

            # Output Layer: num_classes with Softmax
            nn.Linear(32, num_classes),
            nn.Softmax(dim=1)  # Softmax for class probabilities
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
