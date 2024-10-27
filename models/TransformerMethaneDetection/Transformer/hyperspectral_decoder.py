import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=8):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value):
        # Apply multi-head attention and residual connection
        attn_output, _ = self.multihead_attn(query, key, value)
        query = self.norm(query + attn_output)

        # Feed-forward network with residual connection
        ffn_output = self.ffn(query)
        query = query + ffn_output
        query = self.ffn_norm(query)
        return query


class HyperspectralDecoder(nn.Module):
    def __init__(self, d_model=256, n_heads=8, num_layers=1):
        super(HyperspectralDecoder, self).__init__()
        self.layers = nn.ModuleList([
            CrossAttention(d_model, n_heads) for _ in range(num_layers)
        ])

    def forward(self, encoder_output, pos_embeddings, queries):
        # Add positional embeddings to encoder output
        encoder_output = encoder_output + pos_embeddings

        # Pass through each decoder layer with cross-attention
        for layer in self.layers:
            queries = layer(queries, encoder_output, encoder_output)

        return queries



