import torch.nn as nn


class MethaNetClassifier(nn.Module):
    def __init__(self, in_channels:int = 9, num_classes:int = 2):
        super(MethaNetClassifier, self).__init__()
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels//1, in_channels//2, kernel_size=1),
            nn.Conv2d(in_channels//2, in_channels//4, kernel_size=1),
            nn.Conv2d(in_channels//4, in_channels//8, kernel_size=1),
        )

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=2, stride=1, padding=0),  # Input channels = 8
            nn.ReLU(),

            nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(12, 16, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Dropout(p=0.2),

            nn.Conv2d(16, 16, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 61 * 61, 64),  # Adjust dimensions for 512x512 input
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, num_classes),
            nn.Softmax(dim=1)  # Softmax for class probabilities
        )

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
