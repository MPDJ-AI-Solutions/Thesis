import torch
import torch.nn as nn
from models.Transformer.MethaneMapper.Backbone.custom_resnet import CustomResnet

class Backbone(nn.Module):
    """
    Class uses resnet-50 backbones to extract features from input RGB and SWIR images.
    """
    def __init__(self, rgb_channels: int = 3, swir_channels: int = 5, out_channels: int = 1024, d_model: int = 256):
        super(Backbone, self).__init__()
        
        # RGB
        self.rgb_backbone  = CustomResnet(in_channels = rgb_channels, out_channels=out_channels)
        # SWIR
        self.swir_backbone = CustomResnet(in_channels = swir_channels, out_channels=out_channels)
        self.mag1c_backbone = CustomResnet(in_channels = 1, out_channels=out_channels)

        self.combine_projection = nn.Conv2d(in_channels=3*out_channels, out_channels=out_channels, kernel_size=1)

        self.d_model_projection = nn.Conv2d(in_channels=out_channels, out_channels=d_model, kernel_size=1)


    def forward(self, hsi: torch.Tensor, mag1c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Input: Shape(bs, 3, h, w) Output: Shape(bs, out_channels, h / 32, w / 32)
        rgb_result = self.rgb_backbone(self._get_rgb(hsi=hsi))

        # Input: Shape(bs, 5, h, w) Output: Shape(bs, out_channels, h / 32, w / 32)
        swir_result = self.swir_backbone(self._get_swir(hsi=hsi))
        mag1c_result = self.mag1c_backbone(mag1c)

        # (bs, 2 * out_channels, h / 32, w / 32) =  (bs, out_channels, h / 32, w / 32) + (bs, out_channels, h / 32, w / 32)
        combined_result = torch.cat((rgb_result, swir_result, mag1c_result), 1)

        # Input: Shape(bs, 2 * out_channels, h / 32, w / 32) Output: Shape(bs, out_channels, h / 32, w / 32)
        combined_projection = self.combine_projection(combined_result)
        
        # Input: Shape(bs, out_channels, h / 32, w / 32) Output: Shape(bs, d_model, h / 32, w / 32)
        d_model_projection = self.d_model_projection(combined_projection)
        return  d_model_projection, combined_projection,


    @staticmethod
    def _get_rgb(hsi: torch.Tensor) -> torch.Tensor:
        return hsi[:, :3, :, :]


    @staticmethod
    def _get_swir(hsi: torch.Tensor) -> torch.Tensor:
        return hsi[:, 3:8, :, :]
