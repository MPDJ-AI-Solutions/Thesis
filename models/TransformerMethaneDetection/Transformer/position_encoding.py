import torch
import math
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, height, width):
        super(PositionalEncoding, self).__init__()
        pe = self._calculate_matrix_loop(height, width, d_model)

        # Register positional encoding as a buffer (no gradient computation needed)
        self.register_buffer('pe', pe)


    def forward(self, x):
        # x: (batch_size, d_model, height, width)
        batch_size, height, width, d_model = x.size()

        # Expand positional encoding to match batch size and add to input
        pe_expanded = self.pe.unsqueeze(0).expand(batch_size, -1, -1, -1)
        return (x + pe_expanded).view(batch_size, width * height, d_model)

    @staticmethod
    def _calculate_matrix_loop(height, width, d_model):
        pe = torch.zeros(height, width, d_model)
        for h in range(height):
            for w in range(width):
                for ch in range(d_model):
                    if h % 2 == 0:
                        pe[h, w, ch] = math.sin(h / 10000 ** (2 * ch / d_model))
                    else:
                        pe[h, w, ch] = math.cos(h / 10000 ** (2 * ch / d_model))

        for h in range(height):
            for w in range(width):
                for ch in range(d_model):
                    if w % 2 == 0:
                        pe[h, w, ch] += math.sin(w / 10000 ** (2 * ch / d_model))
                    else:
                        pe[h, w, ch] += math.cos(w / 10000 ** (2 * ch / d_model))

        return pe


    @staticmethod
    def _calculate_matrix_tensors(height, width, d_model):
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