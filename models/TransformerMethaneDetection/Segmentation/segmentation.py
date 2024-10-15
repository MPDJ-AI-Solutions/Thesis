import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional

class SegmentationModel(nn.Module):
    def __init__(self, detr, freeze_detr=False):
        super().__init__()
        self.detr = detr

        if freeze_detr:
            for p in self.parameters():
                p.requires_grad_(False)

        hidden_dim, nheads = detr.transformer.d_model, detr.transformer.nhead
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
        self.mask_head = MaskHeadSmallConv(hidden_dim + nheads, [1024, 512, 256], hidden_dim)

        combiner_list = []
        num_channels = detr.backbone.input_size # feature extractor
        for in_channels in range(num_channels):
            combiner_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True)
                )
            )
        self.combiner = nn.ModuleList(combiner_list)

    def forward(self, samples: Tensor, samples1: Tensor, samples2: Tensor):
        # features extraction
        features, pos, mf_out, rgb_feat, raw_feat = self.detr.backbone(
            samples, samples1, samples2
        )

        bs = features[-1].shape[0]
        src = features[-1]
        mf_query = mf_out[-1]

        src_proj = self.detr.input_proj(src)
        hs, memory = self.detr.transformer(src_proj, None, self.detr.query_embed.weight, pos[-1], mf_query)

        # Merge features to create the pyramid
        comb_features = []
        for idx, _layer in enumerate(rgb_feat.keys()):
            _comb_feat = torch.cat((rgb_feat[_layer], raw_feat[_layer]), dim=1)
            _comb_out = self.combiner[idx](_comb_feat)
            comb_features.append(_comb_out)

        outputs_class = self.detr.class_embed(hs)
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.detr.aux_loss:
            out["aux_outputs"] = self.detr._set_aux_loss(outputs_class, outputs_coord)

        bbox_mask = self.bbox_attention(hs[-1], memory)
        seg_masks = self.mask_head(src_proj, bbox_mask, [comb_features[2], comb_features[1], comb_features[0]])
        outputs_seg_masks = seg_masks.view(bs, self.detr.num_queries, seg_masks.shape[-2], seg_masks.shape[-1])

        out["pred_masks"] = outputs_seg_masks
        return out


class MaskHeadSmallConv(nn.Module):
    """Simplified convolutional head with group norm and FPN-based upsampling."""

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8]
        self.lay1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, dim)
        self.lay2 = nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = nn.GroupNorm(8, inter_dims[1])
        self.lay3 = nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = nn.GroupNorm(8, inter_dims[2])
        self.lay4 = nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = nn.GroupNorm(8, inter_dims[3])
        self.out_lay = nn.Conv2d(inter_dims[3], 1, 3, padding=1)

        self.adapter1 = nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

    def forward(self, x: Tensor, bbox_mask: Tensor, fpns: List[Tensor]):
        x = torch.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)

        x = F.relu(self.gn1(self.lay1(x)))
        x = F.relu(self.gn2(self.lay2(x)))

        cur_fpn = self.adapter1(fpns[0])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = F.relu(self.gn3(self.lay3(x)))

        cur_fpn = self.adapter2(fpns[1])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = F.relu(self.gn4(self.lay4(x)))

        x = self.out_lay(x)
        return x


class MHAttentionMap(nn.Module):
    """2D attention module, returns attention softmax (no multiplication)."""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        return F.softmax(weights.flatten(2), dim=-1).view(weights.size())

def _expand(tensor: Tensor, length: int) -> Tensor:
    """
    Expand the tensor along the batch dimension.
    Args:
        tensor (Tensor): The input tensor to be expanded.
        length (int): The number of times to repeat the tensor along the batch dimension.
    Returns:
        Tensor: The expanded tensor.
    """
    return tensor.unsqueeze(1).expand(-1, length, -1, -1, -1).flatten(0, 1)
