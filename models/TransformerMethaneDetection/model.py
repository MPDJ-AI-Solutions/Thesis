import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from SpectralFeatureGenerator.spectral_feature_generator import SpectralFeatureGenerator
from Backbone.backbone import Backbone
from Transformer.encoder import Encoder
from Transformer.hyperspectral_decoder import HyperspectralDecoder
from Transformer.position_encoding import PositionalEncoding
from Transformer.query_refiner import QueryRefiner
from Segmentation.segmentation import BoxAndMaskPredictor


class Model(nn.Module):
    """
    TODO DO IT
    """
    
    def __init__(self):
        self.backbone = Backbone()
        self.spectral_feature_generator = SpectralFeatureGenerator()
        
        self.positional_encoding = PositionalEncoding()
        self.encoder = Encoder()
        
        self.query_refiner = QueryRefiner()
        self.decoder = HyperspectralDecoder()
        
        self.segmentation = BoxAndMaskPredictor()

        
    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        "TODO"
        pass
