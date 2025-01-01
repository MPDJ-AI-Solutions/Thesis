import torch
import torch.nn as nn

from .Classification.classification import ClassifierPredictor
from .Segmentation.bbox_prediction import BBoxPrediction
from .Segmentation.segmentation import BoxAndMaskPredictor
from .SpectralFeatureGenerator.spectral_feature_generator import SpectralFeatureGenerator
from .Backbone.backbone import Backbone
from .Transformer.encoder import Encoder
from .Transformer.hyperspectral_decoder import HyperspectralDecoder
from .Transformer.position_encoding import PositionalEncodingMM
from .Transformer.query_refiner import QueryRefiner
from .model_type import ModelType


class TransformerModel(nn.Module):
    """
    MethaneMapper (Transformer) model for various tasks such as classification, 
    segmentation, and bounding box prediction for methane leakage in hyperspectral images. 

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
                 model_type: ModelType = ModelType.CLASSIFICATION,
                 ):
        """
        Initialize the TransformerModel.
        Args:
            d_model (int): Dimension of the model.
            backbone_out_channels (int): Number of output channels from the backbone.
            image_height (int): Height of the input image.
            image_width (int): Width of the input image.
            attention_heads (int): Number of attention heads.
            n_encoder_layers (int): Number of encoder layers.
            n_decoder_layers (int): Number of decoder layers.
            n_queries (int): Number of queries.
            threshold (float): Threshold value for predictions.
            model_type (ModelType): Type of the model (CLASSIFICATION, SEGMENTATION, ONLY_BBOX).
        """
        super(TransformerModel, self).__init__()

        self.d_model = d_model

        self.backbone = Backbone(d_model=d_model, rgb_channels=3, swir_channels=5, out_channels=backbone_out_channels)
        self.spectral_feature_generator = SpectralFeatureGenerator(d_model=d_model)

        self.positional_encoding = PositionalEncodingMM(
            d_model=d_model
        )
        self.encoder = Encoder(d_model=d_model, n_heads=attention_heads, num_layers=n_encoder_layers)

        self.query_refiner = QueryRefiner(d_model=d_model, num_heads=attention_heads, num_queries=n_queries)
        self.decoder = HyperspectralDecoder(d_model=d_model, n_heads=attention_heads, num_layers=n_decoder_layers)


        self.head = None
        match model_type:
            case ModelType.CLASSIFICATION:
                self.head = ClassifierPredictor(
                    num_classes=2,
                    embedding_dim=d_model,
                )
            case ModelType.SEGMENTATION:
                self.head = BoxAndMaskPredictor(
                    result_width=image_width,
                    result_height=image_height,
                    fpn_channels=backbone_out_channels,
                    embedding_dim=d_model,
                )
            case ModelType.ONLY_BBOX:
                self.head = BBoxPrediction(d_model=d_model)

    def forward(self, image: torch.Tensor, filtered_image: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            image (torch.Tensor): Input hyperspectral image tensor.
            filtered_image (torch.Tensor): SLF result tensor.
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing three tensors as the output of the model.
        """
        # get image size
        batch_size, channels, height, width = image.shape

        f_comb_proj, f_comb = self.backbone(image)

        positional_encoding = self.positional_encoding(f_comb)[0].expand(batch_size, -1, -1, -1)

        f_mc = self.spectral_feature_generator(filtered_image)
        f_mc = f_mc.permute(0, 2, 3, 1)

        q_ref = self.query_refiner(f_mc)
        f_e = self.encoder((f_comb_proj + positional_encoding).flatten(2).permute(0, 2, 1))


        e_out = self.decoder(
            (f_e.permute(0, 2, 1).view(batch_size, -1, int(height / 32), int(width / 32)) + positional_encoding).flatten(2).permute(0, 2, 1),
            q_ref
        )

        result = self.head(e_out, f_e)

        return result