import torch
import torch.nn as nn

from .Segmentation.bbox_prediction import BBoxPrediction
from .SpectralFeatureGenerator.spectral_feature_generator import SpectralFeatureGenerator
from .Backbone.backbone import Backbone
from .Transformer.encoder import Encoder
from .Transformer.hyperspectral_decoder import HyperspectralDecoder
from .Transformer.position_encoding import PositionalEncoding
from .Transformer.query_refiner import QueryRefiner
from .Segmentation.segmentation import BoxAndMaskPredictor


class TransformerModel(nn.Module):
    """
    TODO docs, verification, tests
    """
    
    def __init__(self,
                 d_model: int = 256,
                 backbone_out_channels: int = 2048,
                 image_height: int = 512,
                 image_width: int = 512,
                 attention_heads: int = 8,
                 n_encoder_layers: int = 6,
                 n_decoder_layers: int = 6,
                 n_queries: int = 100,
                 threshold: float = 0.5,
            ):

        super(TransformerModel, self).__init__()

        self.d_model = d_model

        self.backbone = Backbone(d_model=d_model, rgb_channels=3, swir_channels=5, out_channels=backbone_out_channels)
        self.spectral_feature_generator = SpectralFeatureGenerator(d_model=d_model)
        
        self.positional_encoding = PositionalEncoding(
            d_model=d_model, height=int(image_height / 32), width=int(image_width / 32)
        )
        self.encoder = Encoder(d_model=d_model, n_heads=attention_heads, num_layers=n_encoder_layers)
        
        self.query_refiner = QueryRefiner(d_model=d_model, num_heads=attention_heads, num_queries=n_queries)
        self.decoder = HyperspectralDecoder(d_model=d_model, n_heads=attention_heads, num_layers=n_decoder_layers)
        
        # self.segmentation = BoxAndMaskPredictor(
        #     num_heads=attention_heads,
        #     fpn_channels=n_queries,
        #     threshold=threshold,
        #     embedding_dim=d_model,
        #     result_width=image_width,
        #     result_height=image_height,
        # )

        self.bbox = BBoxPrediction(d_model=d_model)
        
    def forward(self, image: torch.Tensor, filtered_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        TODO docs, tests
        """
        # get image size
        batch_size, channels, height, width = image.shape

        f_comb_proj = self.backbone(image)
        f_mc = self.spectral_feature_generator(filtered_image)
        f_mc = f_mc.permute(0, 2, 3, 1)

        q_ref = self.query_refiner(f_mc)
        f_e = self.encoder(self.positional_encoding(f_comb_proj.permute(0, 2, 3, 1)))

        e_out = self.decoder(
            self.positional_encoding(f_e.view(batch_size, int(height / 32), int(width/32), self.d_model)), q_ref
        )

        result = self.bbox(e_out)

        return result
