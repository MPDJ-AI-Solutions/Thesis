from typing import Tuple

import torch
from torch import nn


class BBoxPrediction(nn.Module):
    def __init__(self, d_model):
        super(BBoxPrediction, self).__init__()

        self.bbox_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 5),
            nn.ReLU(),
            nn.Linear(d_model  * 5 , d_model * 8),
            nn.ReLU(),
            nn.Linear(d_model * 8, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, 4),  # Output: bounding box coordinates (x, y, w, h)
        )

        self.class_head = nn.Sequential(
            nn.Linear(d_model, 1),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize layers in bbox_head
        for layer in self.bbox_head:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')  # He initialization
                nn.init.constant_(layer.bias, 0)  # Initialize biases to 0

        # Initialize layers in class_head
        for layer in self.class_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)  # He initialization
                nn.init.constant_(layer.bias, 0)  # Initialize biases to 0


    def forward(self, e_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out_class = self.class_head(e_out)
        bboxes = self.bbox_head(e_out)

        return bboxes.squeeze(0), out_class.squeeze(0)
