import torch
import torch.nn as nn


class BoxAndMaskPredictor(nn.Module):
    """
    BoxAndMaskPredictor is a neural network module designed for predicting bounding boxes, confidence scores, 
    and segmentation masks from input feature maps. It utilizes a combination of feed-forward networks (FFNs) 
    and attention mechanisms to generate these predictions.
    """
    def __init__(
            self, result_height: int,
            result_width: int,
            embedding_dim: int,
            fpn_channels: int,
            num_heads: int = 8,
            threshold: float = 0.5
    ):
        """
            Initializes the BoxAndMaskPredictor.
            
            Args:
                result_height (int): The height of the result, must be divisible by 32.
                result_width (int): The width of the result, must be divisible by 32.
                embedding_dim (int): The dimension of the embedding.
                fpn_channels (int): The number of channels in the Feature Pyramid Network.
                num_heads (int, optional): The number of attention heads. Default is 8.
                threshold (float, optional): The threshold for confidence score. Default is 0.5.
            Raises:
                AssertionError: If result_height or result_width is not divisible by 32.
            """
        
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
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, 4)  # Output: bounding box coordinates (x, y, w, h)
        )

        # FFN for confidence score prediction
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, 1)  # Output: confidence score
        )

        # Attention mechanism for mask prediction
        self.pre_attention_norm = nn.LayerNorm(embedding_dim)
        self.mask_attention = nn.MultiheadAttention(embedding_dim, num_heads=num_heads, batch_first=True)


        self.mask_projection =nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, int((result_height / 32) * (result_width / 32)))
        )

        # Feature Pyramid Network-like layers for upsampling
        self.fpn_layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(fpn_channels, fpn_channels // 2, kernel_size=2, stride=2),
                nn.BatchNorm2d(fpn_channels // 2),
                nn.ReLU(),
                nn.Dropout2d(0.1)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(fpn_channels // 2, fpn_channels // 4, kernel_size=2, stride=2),
                nn.BatchNorm2d(fpn_channels // 4),
                nn.ReLU(),
                nn.Dropout2d(0.1)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(fpn_channels // 4, fpn_channels // 8, kernel_size=2, stride=2),
                nn.BatchNorm2d(fpn_channels // 8),
                nn.ReLU(),
                nn.Dropout2d(0.1)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(fpn_channels // 8, fpn_channels // 16, kernel_size=2, stride=2),
                nn.BatchNorm2d(fpn_channels // 16),
                nn.ReLU(),
                nn.Dropout2d(0.1)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(fpn_channels // 16, 1, kernel_size=2, stride=2),
                nn.BatchNorm2d(1)
            )
        ])

        self._init_weights()

    def _init_weights(self):
        """Initialize the weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, e_out: torch.Tensor, fe: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the segmentation model.
        
        Args:
            e_out (torch.Tensor): Encoder output tensor of shape (batch_size, num_queries, feature_dim).
            fe (torch.Tensor): Feature tensor of shape (batch_size, num_queries, feature_dim).
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - bbox_predictions (torch.Tensor): Bounding box predictions of shape (batch_size, num_queries, 4).
                - confidence_scores (torch.Tensor): Confidence scores of shape (batch_size, num_queries, 1).
                - mask_heatmaps (torch.Tensor): Mask heatmaps of shape (batch_size, fpn_channels, result_height, result_width).
        """
        bbox_predictions = self.bbox_head(e_out)
        bbox_predictions = bbox_predictions * 0.1  # Scale gradients
        bbox_predictions[:, :2] = torch.sigmoid(bbox_predictions[:, :2]) * 512# Clamp x, y to [0, 1]
        bbox_predictions[:, 2:] = torch.relu(bbox_predictions[:, 2:]) * 512 # Clamp w, h to [0, ∞]

        confidence_scores = torch.sigmoid(self.confidence_head(e_out))

        # Compute mask attention scores
        normed_e_out = self.pre_attention_norm(e_out)
        normed_fe = self.pre_attention_norm(fe)
        mask_attention_scores, _ = self.mask_attention(normed_e_out, normed_fe, normed_fe)

        # Project to spatial feature map
        # (batch_size, num_queries, (result_height / 32) * (result_width / 32))
        mask_heatmaps = self.mask_projection(mask_attention_scores)

        # Reshape to fit FPN
        mask_heatmaps = mask_heatmaps.view(-1, self.fpn_channels, int(self.result_height / 32), int(self.result_width / 32))

        # Upsample through FPN
        previous_features = None
        for i, layer in enumerate(self.fpn_layers):
            mask_heatmaps = layer(mask_heatmaps)
            if previous_features is not None and i < len(self.fpn_layers) - 1:
                # Add residual connection if shapes match
                if previous_features.shape == mask_heatmaps.shape:
                    mask_heatmaps = mask_heatmaps + previous_features
            previous_features = mask_heatmaps

        mask_heatmaps = torch.sigmoid(mask_heatmaps / 0.1)  # Temperature scaling

        return bbox_predictions, confidence_scores, mask_heatmaps
