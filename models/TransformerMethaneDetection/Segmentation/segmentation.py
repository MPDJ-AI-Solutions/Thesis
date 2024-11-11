import torch
import torch.nn as nn


class BoxAndMaskPredictor(nn.Module):
    def __init__(
            self, result_height: int,
            result_width: int,
            embedding_dim: int,
            fpn_channels: int,
            num_heads: int = 8,
            threshold: float = 0.5
    ):
        # Ensure result_height and result_width are divisible by 32
        assert result_height % 32 == 0 and result_width % 32 == 0, "result_height and result_width must be divisible by 32"

        super(BoxAndMaskPredictor, self).__init__()

        self.embedding_dim = embedding_dim
        self.fpn_channels = fpn_channels
        self.threshold = threshold

        self.result_height = result_height
        self.result_width = result_width

        # FFNs for box prediction
        self.bbox_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 4)  # Output: bounding box coordinates (x, y, w, h)
        )

        # FFN for confidence score prediction
        self.confidence_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)  # Output: confidence score
        )

        # Attention mechanism for mask prediction
        self.mask_attention = nn.MultiheadAttention(embedding_dim, num_heads=num_heads, batch_first=True)


        self.mask_projection = nn.Linear(embedding_dim, int((result_height / 32) * (result_width / 32)))

        # Feature Pyramid Network-like layers for upsampling
        self.fpn_layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(fpn_channels, fpn_channels // 2, kernel_size=2, stride=2),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(fpn_channels // 2, fpn_channels // 4, kernel_size=2, stride=2),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(fpn_channels // 4, fpn_channels // 8, kernel_size=2, stride=2),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(fpn_channels // 8, fpn_channels // 16, kernel_size=2, stride=2),
                nn.ReLU()
            ),
            nn.ConvTranspose2d(fpn_channels // 16, 1, kernel_size=2, stride=2)  # Final output: 1 channel
        ])

    def forward(self, e_out: torch.Tensor, fe: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bbox_predictions = self.bbox_head(e_out)
        confidence_scores = torch.sigmoid(self.confidence_head(e_out))

        # Compute mask attention scores
        mask_attention_scores, _ = self.mask_attention(e_out, fe, fe)

        # Project to spatial feature map
        # (batch_size, num_queries, (result_height / 32) * (result_width / 32))
        mask_heatmaps = self.mask_projection(mask_attention_scores)

        # Reshape to fit FPN
        mask_heatmaps = mask_heatmaps.view(-1, self.fpn_channels, int(self.result_height / 32), int(self.result_width / 32))

        # Upsample through FPN
        for layer in self.fpn_layers:
            mask_heatmaps = layer(mask_heatmaps)

        mask_heatmaps = torch.sigmoid(mask_heatmaps)

        return bbox_predictions, confidence_scores, mask_heatmaps
