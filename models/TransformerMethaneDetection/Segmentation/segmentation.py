import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from collections import defaultdict

class SegmentationModel(nn.Module):
    """
    Segmentation model with Transformer-based mask prediction and a convolutional mask head.
    """
    def __init__(self, backbone_model, freeze_backbone=False):
        super().__init__()
        self.backbone = backbone_model

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden_dim = backbone_model.transformer.hidden_dim
        num_heads = backbone_model.transformer.num_heads
        
        # Attention layer to highlight bounding box areas
        self.bbox_attention = AttentionMap(hidden_dim, hidden_dim, num_heads, dropout=0.0)
        
        # Mask prediction head with downsampling convolutional layers
        self.mask_head = MaskPredictionHead(hidden_dim + num_heads, [1024, 512, 256], hidden_dim)

        # Combining RGB and raw features from the backbone at multiple scales
        self.feature_combiner = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1),
                nn.ReLU(inplace=True)
            )
            for in_channels in backbone_model.feature_channels
        ])

    def forward(self, rgb_input, raw_input):
        rgb_features, raw_features, mask_features, positional_encoding = self.backbone(rgb_input, raw_input)

        # Merge RGB and raw feature layers for pyramid structure
        combined_features = []
        for rgb_layer, raw_layer, comb_layer in zip(rgb_features, raw_features, self.feature_combiner):
            combined_features.append(comb_layer(torch.cat([rgb_layer, raw_layer], dim=1)))

        # Transformer-based object detection outputs
        hidden_states, memory = self.backbone.transformer(mask_features)
        class_logits = self.backbone.class_head(hidden_states)
        box_predictions = self.backbone.bbox_head(hidden_states).sigmoid()

        output = {
            "class_logits": class_logits[-1],
            "boxes": box_predictions[-1],
        }

        # Compute bounding box mask for segmentation
        bbox_mask = self.bbox_attention(hidden_states[-1], memory)
        segmentation_masks = self.mask_head(mask_features, bbox_mask, combined_features)
        
        # Format segmentation masks
        output["masks"] = segmentation_masks.view(
            mask_features.size(0), self.backbone.num_queries, segmentation_masks.size(-2), segmentation_masks.size(-1)
        )
        return output


class AttentionMap(nn.Module):
    """
    Computes attention maps for bounding box refinement.
    """
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.query_transform = nn.Linear(input_dim, hidden_dim)
        self.key_transform = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)

    def forward(self, query, key):
        query = self.query_transform(query)
        key = self.key_transform(key)
        
        query_heads = query.view(query.shape[0], query.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        key_heads = key.view(key.shape[0], self.num_heads, self.hidden_dim // self.num_heads, key.shape[-2], key.shape[-1])
        
        attention_weights = torch.einsum("bqnc,bnchw->bqnhw", query_heads, key_heads) / (self.hidden_dim ** 0.5)
        attention_weights = F.softmax(attention_weights.view(attention_weights.size(0), attention_weights.size(1), -1), dim=-1)
        
        return self.dropout(attention_weights.view_as(attention_weights))


class MaskPredictionHead(nn.Module):
    """
    Convolutional head for mask prediction, using feature pyramid networks (FPN) for upsampling.
    """
    def __init__(self, input_dim, pyramid_dims, context_dim):
        super().__init__()
        self.input_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_dim, context_dim // (2 ** i), kernel_size=3, padding=1),
                nn.GroupNorm(8, context_dim // (2 ** i)),
                nn.ReLU(inplace=True)
            ) for i in range(5)
        ])
        
        self.output_layer = nn.Conv2d(context_dim // 16, 1, kernel_size=3, padding=1)
        self.adapters = nn.ModuleList([
            nn.Conv2d(pyramid_dim, context_dim // (2 ** i), kernel_size=1) 
            for i, pyramid_dim in enumerate(pyramid_dims)
        ])

    def forward(self, input_tensor, bbox_mask, feature_pyramids):
        x = torch.cat([input_tensor, bbox_mask], dim=1)

        for layer, fpn, adapter in zip(self.input_layers, feature_pyramids, self.adapters):
            x = layer(x)
            upsampled_fpn = adapter(fpn)
            x += F.interpolate(upsampled_fpn, size=x.shape[-2:], mode="nearest")

        return self.output_layer(x)


class DiceLoss(nn.Module):
    """
    Compute Dice loss for binary masks.
    """
    def forward(self, predictions, targets, num_samples):
        predictions = predictions.sigmoid().flatten(1)
        intersection = 2 * (predictions * targets).sum(dim=1)
        union = predictions.sum(dim=1) + targets.sum(dim=1)
        return (1 - (intersection + 1) / (union + 1)).sum() / num_samples


class FocalLoss(nn.Module):
    """
    Compute Focal loss for segmentation masks, balancing easy vs. hard examples.
    """
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, targets, num_samples):
        prob = predictions.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss *= alpha_factor
        return loss.mean() / num_samples
