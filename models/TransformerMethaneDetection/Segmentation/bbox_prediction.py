from typing import Tuple

import torch
from torch import nn


class BBoxPrediction(nn.Module):
    def __init__(self, d_model):
        super(BBoxPrediction, self).__init__()

        self.bbox_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4),  # Output: bounding box coordinates (x, y, w, h)
            nn.Sigmoid()
        )

        self.class_head = nn.Sequential(
            nn.Linear(d_model, 2),
            nn.Sigmoid()
        )


    def forward(self, e_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out_class = self.class_head(e_out)
        bboxes = self.bbox_head(e_out)

        return bboxes, out_class
