import torch
import torch.nn as nn
from models.Transformer.MethaneMapper.Backbone.custom_resnet import CustomResnet

class Backbone(nn.Module):
    """
    Class uses resnet-50 backbones to extract features from input RGB and SWIR images.
    """
    def __init__(self, rgb_channels: int = 3, swir_channels: int = 5, out_channels: int = 1024, d_model: int = 256):
        """
        Initializes the Backbone model.
        Args:
            rgb_channels (int): Number of input channels for the RGB backbone. Default is 3.
            swir_channels (int): Number of input channels for the SWIR backbone. Default is 5.
            out_channels (int): Number of output channels for the backbones and combined projection. Default is 1024.
            d_model (int): Number of output channels for the final projection. Default is 256.
        """
        super(Backbone, self).__init__()
        
        # RGB
        self.rgb_backbone  = CustomResnet(in_channels = rgb_channels, out_channels=out_channels)
        # SWIR
        self.swir_backbone = CustomResnet(in_channels = swir_channels, out_channels=out_channels)
        self.mag1c_backbone = CustomResnet(in_channels = 1, out_channels=out_channels)

        self.combine_projection = nn.Conv2d(in_channels=3*out_channels, out_channels=out_channels, kernel_size=1)

        self.d_model_projection = nn.Conv2d(in_channels=out_channels, out_channels=d_model, kernel_size=1)


    def forward(self, hsi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            hsi (torch.Tensor): Input tensor with hyperspectral images. 
        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - d_model_projection (torch.Tensor): Tensor with shape (bs, d_model, h / 32, w / 32).
                - combined_projection (torch.Tensor): Tensor with shape (bs, out_channels, h / 32, w / 32).
        """
        # Input: Shape(bs, 3, h, w) Output: Shape(bs, out_channels, h / 32, w / 32)
        rgb_result = self.rgb_backbone(self._get_rgb(hsi=hsi))

        # Input: Shape(bs, 5, h, w) Output: Shape(bs, out_channels, h / 32, w / 32)
        swir_result = self.swir_backbone(self._get_swir(hsi=hsi))
        mag1c_result = self.mag1c_backbone(self._get_mag1c(hsi=hsi))

        # (bs, 2 * out_channels, h / 32, w / 32) =  (bs, out_channels, h / 32, w / 32) + (bs, out_channels, h / 32, w / 32)
        combined_result = torch.cat((rgb_result, swir_result, mag1c_result), 1)

        # Input: Shape(bs, 2 * out_channels, h / 32, w / 32) Output: Shape(bs, out_channels, h / 32, w / 32)
        combined_projection = self.combine_projection(combined_result)
        
        # Input: Shape(bs, out_channels, h / 32, w / 32) Output: Shape(bs, d_model, h / 32, w / 32)
        d_model_projection = self.d_model_projection(combined_projection)
        return  d_model_projection, combined_projection,


    @staticmethod
    def _get_rgb(hsi: torch.Tensor) -> torch.Tensor:
        """
        Extracts the RGB channels from an HSI (Hyperspectral Imaging) tensor.

        Args:
            hsi (torch.Tensor): A tensor containing hyperspectral image data with shape 
                                (batch_size, channels, height, width).

        Returns:
            torch.Tensor: A tensor containing only the RGB channels with shape 
                          (batch_size, 3, height, width).
        """
        return hsi[:, 5:8, :, :]


    @staticmethod
    def _get_swir(hsi: torch.Tensor) -> torch.Tensor:
        """
        Extracts the Short-Wave Infrared (SWIR) bands from a hyperspectral image tensor.

        Args:
            hsi (torch.Tensor): A hyperspectral image tensor with shape (batch_size, num_channels, height, width).

        Returns:
            torch.Tensor: A tensor containing the SWIR bands (channels 3 to 7) of the input hyperspectral image.
        """
        return hsi[:, :5, :, :]

    @staticmethod
    def _get_mag1c(hsi: torch.Tensor) -> torch.Tensor:
        """
        Extracts mag1c from the input hyperspectral image tensor.

        Args:
            hsi (torch.Tensor): A hyperspectral image tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: A tensor containing the extracted channel, with shape (batch_size, 1, height, width).
        """
        return hsi[:, 8:9, :, :]

