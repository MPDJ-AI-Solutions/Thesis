import torch
from torch import nn

class SelfAttentionLayer(nn.Module):
    """
    TODO: COMMENT
    """
    def __init__(self, d_model: int = 256, n_head: int = 8, dropout: int = 0.1):
        super(SelfAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.attention_norm = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        src = x
        x, _ = self.multihead_attn(x, x, x)
        x = self.dropout1(x)
        x = self.attention_norm(x + src)
        src = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.ffn_norm(x + src)
        return x


class Encoder(nn.Module):
    """
    TODO: COMMENT
    """
    def __init__(self, num_layers: int = 1, d_model: int = 256, n_heads: int = 8):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [SelfAttentionLayer(d_model, n_heads) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        return x


