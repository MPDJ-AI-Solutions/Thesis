import torch
import torch.nn as nn

class QueryRefiner(nn.Module):
    """
    QueryRefiner is a neural network module designed to refine queries using self-attention and cross-attention mechanisms.

    Attributes:
        queries (nn.Parameter): Learnable queries of shape (num_queries, d_model).
        self_attention (nn.MultiheadAttention): Self-attention layer with specified embed_dim and num_heads.
        cross_attention (nn.MultiheadAttention): Cross-attention layer with specified embed_dim and num_heads.
        layer_norm (nn.LayerNorm): Layer normalization applied after self-attention.
    """

    def __init__(self, d_model = 256, num_queries=100, num_heads=8):
        """
        Initializes the QueryRefiner module.

        Args:
            d_model (int, optional): The dimension of the model. Default is 256.
            num_queries (int, optional): The number of learnable queries. Default is 100.
            num_heads (int, optional): The number of attention heads. Default is 8.
    
        """
        super(QueryRefiner, self).__init__()

        # Learnable queries
        # 100 x 256
        self.queries = nn.Parameter(torch.randn(num_queries, d_model))

        # Self-attention and cross-attention layers
        # d = 256, heads = 8
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, f_mc: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass for the query refiner module.
        
        Args:
            f_mc (torch.Tensor): Input feature map tensor of shape (batch_size, H, W, d_model).
        Returns:
            torch.Tensor: Refined queries tensor after self-attention and cross-attention.
        """
        # Self-attention on queries
        batch_size, H, W, d_model = f_mc.shape
        queries = self.queries.repeat(batch_size, 1, 1)

        q_self_att, _ = self.self_attention(queries, queries, queries)

        # Layer Norm (queries + Self-Attention)
        q_self_att_normalized = self.layer_norm(self.queries + q_self_att)

        # Reshape fmc (feature map) from (bs, H, W, d) to (bs, H*W, d) for compatibility
        f_mc_flattened = f_mc.view(batch_size, H * W, d_model)

        # Cross-attention between queries and methane candidate feature map
        q_ref, _ = self.cross_attention(q_self_att_normalized, f_mc_flattened, f_mc_flattened)

        return q_ref
