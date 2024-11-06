import torch
import torch.nn as nn
import torch.nn.functional as F

class BoxAndMaskPredictor(nn.Module):
    def __init__(self, embedding_dim: int, fpn_channels: int, num_heads: int = 8, threshold: float = 0.5):
        super(BoxAndMaskPredictor, self).__init__()
        
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
        
        # Multi-head attention for mask prediction
        self.mask_attention = nn.MultiheadAttention(embedding_dim, num_heads=num_heads)
        
        # Feature Pyramid Network-like structure for upsampling
        self.fpn_layers = nn.ModuleList([
            nn.ConvTranspose2d(fpn_channels,      fpn_channels // 2, kernel_size=2, stride=2),
            nn.ConvTranspose2d(fpn_channels // 2, fpn_channels // 4, kernel_size=2, stride=2),
            nn.ConvTranspose2d(fpn_channels // 4,                 1, kernel_size=2, stride=2)
        ])
        
        # Threshold for generating final segmentation mask
        self.threshold = threshold

    def forward(self, e_out: torch.Tensor, fe: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Predict bounding boxes and confidence scores
        bbox_predictions = self.bbox_head(e_out)  # Shape: (batch_size, num_queries, 4)
        confidence_scores = torch.sigmoid(self.confidence_head(e_out))  # Shape: (batch_size, num_queries, 1)
        
        # Compute attention scores and heatmaps for mask prediction
        mask_attention_scores, _ = self.mask_attention(e_out.permute(1, 0, 2), fe.permute(1, 0, 2), fe.permute(1, 0, 2))
        mask_heatmaps = mask_attention_scores.mean(dim=0).view(-1, fe.shape[2], fe.shape[3])  # Averaging across heads
        
        # Upsample the mask heatmaps using FPN
        for layer in self.fpn_layers:
            mask_heatmaps = layer(mask_heatmaps)

        # Apply thresholding to get the final segmentation mask
        final_mask = (mask_heatmaps > self.threshold).float()
        
        return bbox_predictions, confidence_scores, final_mask
