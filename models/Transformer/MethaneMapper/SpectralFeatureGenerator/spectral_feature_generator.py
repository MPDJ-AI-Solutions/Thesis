import torch
import torch.nn as nn

from models.Transformer.MethaneMapper.SpectralFeatureGenerator.feature_extractor import FeatureExtractor

class SpectralFeatureGenerator(nn.Module):
    """
    Feature generator - class combines usage of spectral linear filter and feature extractor(resnet50 model).
    """

    def __init__(self, d_model: int = 256):
        """
        Initializes the SpectralFeatureGenerator.

        Args:
            d_model (int): The dimension of the model. Default is 256.
        """
        super(SpectralFeatureGenerator, self).__init__()
        self.feature_extractor = FeatureExtractor(d_model=d_model)


    def forward(self, filtered_hyperspectral_image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SpectralFeatureGenerator model.
        
        Args:
            filtered_hyperspectral_image (torch.Tensor): The input hyperspectral image tensor 
                with shape (batch_size, 1, width, height).
        Returns:
            torch.Tensor: The output tensor after feature extraction with shape 
                (batch_size, d_model, height/32, width/32).
        """
        methane_pattern = [0, 0, 0, 0.1, 0.3, 0.6, 0.8, 0.7]

        # Input: Shape(bs, 1, w, h) Output: Shape (bs, d_model, h/32, w/32)
        image = self.feature_extractor(filtered_hyperspectral_image)
        
        return image