import torch
import torch.nn as nn

from models.TransformerMethaneDetection.Backbone.backbone import Backbone
from models.TransformerMethaneDetection.Segmentation.segmentation import SegmentationModel
from models.TransformerMethaneDetection.Transformer.position_encoding import PositionalEncoding
from models.TransformerMethaneDetection.Transformer.transformer import Transformer


class Detector(nn.Module):
    def __init__(self, backbone: Backbone, transformer: Transformer, num_queries=100, d_model=512):
        super(Detector, self).__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.pos_encoder = PositionalEncoding()
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.mf_query = nn.Parameter(torch.rand(num_queries, d_model))

    def forward(self, x, mask):
        """
        Forward pass through the network. It takes batch of spectral images as input.

        """

        features = self.backbone(x)
        pos_embed = self.pos_encoder(features)


        transformer_output, memory = self.transformer(features, mask, pos_embed, self.query_embed.weight, self.mf_query)

        return transformer_output, memory