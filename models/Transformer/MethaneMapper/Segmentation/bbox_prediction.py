import torch

from torch import nn
from typing import Tuple


class BBoxPrediction(nn.Module):
    """
    A PyTorch module for bounding box prediction and classification.
    This module consists of two heads:
    1. bbox_head: A sequential neural network for predicting bounding box coordinates (x, y, w, h).
    2. class_head: A linear layer for predicting the class of the bounding box.
    """

    def __init__(self, d_model):
        """
        Initializes the BBoxPrediction module.
        Args:
            d_model (int): The dimension of the model.
        """
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
        """
        Initialize the weights of the layers in bbox_head and class_head.
        For bbox_head:
        - If the layer is an instance of nn.Linear, initialize the weights using He initialization (kaiming_normal_)
          with 'fan_out' mode and 'relu' nonlinearity.
        - Initialize the biases to 0.
        For class_head:
        - If the layer is an instance of nn.Linear, initialize the weights using Xavier uniform initialization (xavier_uniform_).
        - Initialize the biases to 0.
        """
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
        """
        Forward pass for the bounding box prediction model.
        Args:
            e_out (torch.Tensor): The encoded output from the transformer model.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - bboxes (torch.Tensor): The predicted bounding boxes.
                - out_class (torch.Tensor): The predicted class scores.
        """
        out_class = self.class_head(e_out)
        bboxes = self.bbox_head(e_out)

        return bboxes.squeeze(0), out_class.squeeze(0)
