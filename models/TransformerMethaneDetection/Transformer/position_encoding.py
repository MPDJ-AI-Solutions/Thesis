import torch
import math
import torch.nn as nn


class PositionalEncodingMM(nn.Module):
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
        # x = tensor_list.tensors
        # mask = tensor_list.mask
        b, ch, h, w = x.shape
        not_mask = torch.ones((b, h, w), dtype=bool, device=x.device)
        # assert mask is not None
        # not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos, ~not_mask


class PositionalEncoding(nn.Module):
    """
    TODO: COMMENT
    """
    def __init__(self, d_model: int, height: int, width: int):
        super(PositionalEncoding, self).__init__()
        pe = self._calculate_matrix_tensors(height, width, d_model)

        # Register positional encoding as a buffer (no gradient computation needed)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, d_model, height, width)
        batch_size, height, width, d_model  = x.size()

        # Expand positional encoding to match batch size and add to input
        pe_expanded = self.pe.unsqueeze(0).expand(batch_size, -1, -1, -1)
        return (x + pe_expanded).reshape(batch_size, width * height, d_model)


    @staticmethod
    def _calculate_matrix_tensors(height: int, width: int, d_model: int) -> torch.Tensor:
        # Create a positional encoding matrix of shape (d_model, height, width)
        pe = torch.zeros(d_model, height, width)

        # Generate y and x positions
        y_position = torch.arange(0, height, dtype=torch.float).unsqueeze(1)  # Shape: (height, 1)
        x_position = torch.arange(0, width, dtype=torch.float).unsqueeze(1)  # Shape: (width, 1)

        # Calculate div_term for each channel
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine and cosine encoding for the height (y) dimension across all channels
        pe[0::2, :, :] = torch.sin(y_position * div_term).transpose(0, 1).unsqueeze(2).expand(-1, height, width)
        pe[1::2, :, :] = torch.cos(y_position * div_term).transpose(0, 1).unsqueeze(2).expand(-1, height, width)

        # Apply sine and cosine encoding for the width (x) dimension across all channels
        pe[0::2, :, :] += torch.sin(x_position * div_term).transpose(0, 1).unsqueeze(1).expand(-1, height, width)
        pe[1::2, :, :] += torch.cos(x_position * div_term).transpose(0, 1).unsqueeze(1).expand(-1, height, width)

        return pe.permute(1, 2, 0)
