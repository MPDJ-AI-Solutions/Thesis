import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


# 3.4 Spectral Feature Generator (SFG)
## MP
class SpectralLinearFilter(nn.Module):
    def __init__(self, spectral_absorption_signature, class_mean, class_cov):
        super(SpectralLinearFilter, self).__init__()
        self.spectral_absorption_signature = spectral_absorption_signature
        self.class_mean = class_mean
        self.class_cov = class_cov

    def forward(self, x):
        # Perform per-class covariance-based linear filtering
        class_mean_expanded = self.class_mean.view(1, -1, 1, 1)
        diff = x - class_mean_expanded

        filtered_output = torch.matmul(torch.matmul(diff, torch.inverse(self.class_cov)),
                                       self.spectral_absorption_signature)
        return filtered_output


## DJ
class FeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super(FeatureExtractor, self).__init__()
        self.backbone = backbone  # e.g., ResNet-50

    def forward(self, x):
        return self.backbone(x)


## DJ
class SpectralFeatureGenerator(nn.Module):
    def __init__(self, slf, feature_extractor):
        super(SpectralFeatureGenerator, self).__init__()
        self.slf = slf
        self.feature_extractor = feature_extractor

    def forward(self, hsi):
        ch4_candidates = self.slf(hsi)
        fmc = self.feature_extractor(ch4_candidates)
        return fmc


# 3.5 Query Refiner (QR)
## MP
class QueryRefiner(nn.Module):
    def __init__(self, num_queries, d_model):
        super(QueryRefiner, self).__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, d_model))
        self.self_attention = nn.MultiheadAttention(d_model, num_heads=8)
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads=8)

    def forward(self, fmc):
        q = self.queries.unsqueeze(1)  # Add batch dimension
        q = self.self_attention(q, q, q)[0]  # Self-attention on queries
        q_ref = self.cross_attention(q, fmc, fmc)[0]  # Cross-attention with fmc
        return q_ref


# 3.6 Hyperspectral Decoder
## MP
class HyperspectralDecoder(nn.Module):
    def __init__(self, d_model):
        super(HyperspectralDecoder, self).__init__()
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads=8)

    def forward(self, fe, q_ref):
        e_out = self.cross_attention(q_ref, fe, fe)[0]
        return e_out


# 3.7 Box and Mask Prediction
## DJ
class BoxAndMaskPredictor(nn.Module):
    def __init__(self, d_model, num_queries):
        super(BoxAndMaskPredictor, self).__init__()
        self.bbox_head = nn.Linear(d_model, 4)  # Bounding box output
        self.confidence_head = nn.Linear(d_model, 1)  # Confidence score output
        self.mask_head = nn.Conv2d(256, 1, kernel_size=1)  # Reduce channels from 256 to 1
        self.conv_transpose = nn.ConvTranspose2d(1, 1, kernel_size=(1024, 2), stride=(1, 2), padding=(0, 0))

    def forward(self, e_out, fe, f_comb):
        bbox = self.bbox_head(e_out)
        confidence = self.confidence_head(e_out)
        mask = self.mask_head(fe)
        mask = self.conv_transpose(mask)
        return bbox, confidence, mask


class MethaneMapper(nn.Module):
    def __init__(self, backbone_rgb, backbone_swir, backbone_extractor, d_model, num_queries, slf_params, height, width):
        super(MethaneMapper, self).__init__()
        self.backbone_rgb = backbone_rgb
        self.backbone_swir = backbone_swir
        self.concat_conv = nn.Conv2d(1024, 512, kernel_size=1)  # Reduce to N=2048

        # Positional embedding should have the same channel size as the combined feature map (2048 channels)
        self.positional_embedding = nn.Parameter(torch.randn(1, 512, height // 32, width // 32))

        self.encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=8)

        slf = SpectralLinearFilter(*slf_params)
        self.sfg = SpectralFeatureGenerator(slf, FeatureExtractor(backbone_extractor))
        self.qr = QueryRefiner(num_queries, d_model)
        self.decoder = HyperspectralDecoder(d_model)
        self.box_mask_predictor = BoxAndMaskPredictor(d_model, num_queries)
        self.projection = nn.Conv2d(512, d_model, kernel_size=1)

    def forward(self, hsi):
        # Input is already an 8-channel image (R, G, B + 5 SWIR bands)
        x_rgb = hsi[:, :3]  # First 3 channels for RGB
        x_swir = hsi[:, 3:]  # Remaining 5 channels for SWIR bands

        f_rgb = self.backbone_rgb(x_rgb)  # Apply RGB backbone
        f_swir = self.backbone_swir(x_swir)  # Apply SWIR backbone

        print(f"f_rgb shape: {f_rgb.shape}, f_swir shape: {f_swir.shape}")  # Check the shapes

        f_comb = self.concat_conv(torch.cat([f_rgb, f_swir], dim=1))  # Concatenate and reduce channels
        print(f"f_comb shape after concat and conv: {f_comb.shape}")  # Check f_comb shape

        # Broadcast positional embedding to match batch size
        pos_embedding = F.interpolate(self.positional_embedding, size=(f_comb.shape[2], f_comb.shape[3]), mode='bilinear', align_corners=False)

        print(f"pos_embedding shape: {pos_embedding.shape}")  # Check positional embedding shape
        f_comb = f_comb + pos_embedding

        # Project the feature map to match the transformer's embedding dimension (512)
        f_comb_projected = self.projection(f_comb)
        print(f"f_comb_projected shape after projection: {f_comb_projected.shape}")
        f_comb_pooled = F.adaptive_avg_pool2d(f_comb_projected, (16, 16))  # Reduce spatial dimensions to 16x16
        print(f"f_comb_pooled shape after adaptive pooling: {f_comb_pooled.shape}")

        # Flatten spatial dimensions and permute for transformer input
        f_z = self.encoder(f_comb_pooled.flatten(2).permute(2, 0, 1))  # Fl

        # Spectral Feature Generator and Query Refiner
        fmc = self.sfg(hsi).flatten(1).permute(1, 0).unsqueeze(1)
        q_ref = self.qr(fmc)

        # Decoder and mask prediction
        e_out = self.decoder(f_z, q_ref)
        bbox, confidence, mask = self.box_mask_predictor(e_out, f_z, f_comb)

        return bbox, confidence, mask


# Example instantiation
backbone_rgb = nn.Conv2d(3, 512, kernel_size=3, padding=1)  # Placeholder backbone
backbone_swir = nn.Conv2d(5, 512, kernel_size=3, padding=1)  # Placeholder backbone
backbone_extractor = nn.Conv2d(1, 512, kernel_size=3, padding=1)
slf_params = (torch.randn(256), torch.randn(8), torch.randn(256, 256))  # Placeholder SLF params

# Specify image height and width (e.g., 256x256)
model = MethaneMapper(backbone_rgb, backbone_swir, backbone_extractor, d_model=512, num_queries=100, slf_params=slf_params, height=256,
                      width=256)

# Example input (batch_size=1, channels=8, height=256, width=256)
hsi = torch.randn(1, 8, 256, 256)
bbox, confidence, mask = model(hsi)

cv2.imshow("TEST", mask.detach().squeeze().numpy())
cv2.waitKey(0)

print(bbox.shape, confidence.shape, mask.shape)