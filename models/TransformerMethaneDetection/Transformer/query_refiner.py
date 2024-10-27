import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryRefiner(nn.Module):
    def __init__(self, d_model = 256, num_queries=100, num_heads=8):
        super(QueryRefiner, self).__init__()

        # Learnable queries
        # 100 x 256
        self.queries = nn.Parameter(torch.randn(num_queries, d_model))

        # Self-attention and cross-attention layers
        # d = 256, heads = 8
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, f_mc):
        # Self-attention on queries
        q_self_att, _ = self.self_attention(self.queries.unsqueeze(1), self.queries.unsqueeze(1),
                                            self.queries.unsqueeze(1))

        # Layer Norm (queries + Self-Attention)
        q_self_att_normalized = self.layer_norm(self.queries + q_self_att.squeeze(1))

        # Reshape fmc (feature map) from (H, W, d) to (H*W, d) for compatibility
        H, W, d = f_mc.shape
        f_mc_flattened = f_mc.view(H * W, d)
        f_mc_flattened = f_mc_flattened.unsqueeze(1)  # (H*W, 1, d) for attention

        # Reshape q_self_att_normalized for cross attention
        q_self_att_normalized = q_self_att_normalized.unsqueeze(1) # (num_queries, 1, d_model) for cross attention

        # Cross-attention between queries and methane candidate feature map
        q_ref, _ = self.cross_attention(q_self_att_normalized, f_mc_flattened, f_mc_flattened)

        return q_ref
