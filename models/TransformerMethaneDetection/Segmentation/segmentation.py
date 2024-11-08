import torch
import torch.nn as nn

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
        self.mask_attention = nn.MultiheadAttention(embedding_dim, num_heads=num_heads, batch_first=True)
        
        # Feature Pyramid Network-like structure for upsampling
        self.fpn_layers = nn.ModuleList([
            nn.ConvTranspose2d(fpn_channels, fpn_channels // 2, kernel_size=2, stride=2),
            nn.ConvTranspose2d(fpn_channels // 2, fpn_channels // 4, kernel_size=2, stride=2),
            nn.ConvTranspose2d(fpn_channels // 4, 1, kernel_size=2, stride=2)  # Output 1 channel for binary mask
        ])
        
        # Threshold for generating final segmentation mask
        self.threshold = threshold

    def forward(self, e_out: torch.Tensor, fe: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bbox_predictions = self.bbox_head(e_out) 
        confidence_scores = torch.sigmoid(self.confidence_head(e_out))  
        
        mask_attention_scores, _ = self.mask_attention(e_out, fe, fe)  
        mask_heatmaps = mask_attention_scores.mean(dim=1)  
        
        spatial_dim = int(mask_heatmaps.size(1) ** 0.5)
        mask_heatmaps = mask_heatmaps.view(-1, spatial_dim, spatial_dim)  

        for layer in self.fpn_layers:
            mask_heatmaps = layer(mask_heatmaps) 

        final_mask = (mask_heatmaps > self.threshold).float()  
        
        return bbox_predictions, confidence_scores, final_mask
