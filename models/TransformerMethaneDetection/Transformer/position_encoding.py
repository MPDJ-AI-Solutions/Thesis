import torch
import math
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, d_model=128, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = d_model / 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        # x is the input tensor
        b, ch, h, w = x.shape

        # Create cumulative sums for position embedding
        y_embed = torch.arange(h, dtype=torch.float32, device=x.device).unsqueeze(1).repeat(1, w)
        x_embed = torch.arange(w, dtype=torch.float32, device=x.device).unsqueeze(0).repeat(h, 1)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=2).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=2).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1).unsqueeze(0)
        return pos