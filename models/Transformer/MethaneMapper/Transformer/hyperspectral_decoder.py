import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    """
    CrossAttention module for applying multi-head attention and feed-forward network with residual connections.
    Args:
        d_model (int): The dimension of the model. Default is 256.
        n_heads (int): The number of attention heads. Default is 8.
    """

    def __init__(self, d_model=256, n_heads=8):
        """
        Initializes the CrossAttention module.

        Args:
            d_model (int, optional): The dimension of the model. Default is 256.
            n_heads (int, optional): The number of attention heads. Default is 8.
        """
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the hyperspectral decoder.
        
        Args:
            query (torch.Tensor): The input tensor for the query.
            key (torch.Tensor): The input tensor for the key.
            value (torch.Tensor): The input tensor for the value.
        Returns:
            torch.Tensor: The output tensor after applying multi-head attention,
                          normalization, and feed-forward network.
        """
        # Apply multi-head attention and residual connection
        attn_output, _ = self.multihead_attn(query, key, value)
        query = self.norm(query + attn_output)

        # Feed-forward network with residual connection
        ffn_output = self.ffn(query)
        query = query + ffn_output
        query = self.ffn_norm(query)
        return query


class HyperspectralDecoder(nn.Module):
    """
    A decoder module for hyperspectral data using cross-attention mechanism.
    """

    def __init__(self, d_model=256, n_heads=8, num_layers=1):
        """
        Initializes the HyperspectralDecoder.

        Args:
            d_model (int, optional): The dimension of the model. Default is 256.
            n_heads (int, optional): The number of attention heads. Default is 8.
            num_layers (int, optional): The number of layers in the decoder. Default is 1.
        """
        super(HyperspectralDecoder, self).__init__()
        self.layers = nn.ModuleList([
            CrossAttention(d_model, n_heads) for _ in range(num_layers)
        ])

    def forward(self, encoder_output: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hyperspectral decoder.
        
        Args:
            encoder_output (torch.Tensor): The output tensor from the encoder with shape (batch_size, seq_len, d_model).
            queries (torch.Tensor): The input queries tensor with shape (batch_size, seq_len, d_model).
        Returns:
            torch.Tensor: The output tensor after passing through the decoder layers with shape (batch_size, seq_len, d_model).
        """
        # Pass through each decoder layer with cross-attention
        for layer in self.layers:
            queries = layer(queries, encoder_output, encoder_output)

        return queries



