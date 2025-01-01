import torch
from torch import nn

class SelfAttentionLayer(nn.Module):
    """
    Self-Attention Layer for Transformer model.
    This layer consists of a multi-head self-attention mechanism followed by a feed-forward neural network.
    Layer normalization and dropout are applied after each sub-layer.
    Args:
        d_model (int): The dimension of the input embeddings. Default is 256.
        n_head (int): The number of attention heads. Default is 8.
        dropout (float): The dropout rate. Default is 0.1.
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Applies the self-attention mechanism and feed-forward network to the input tensor.
    Attributes:
        multihead_attn (nn.MultiheadAttention): Multi-head attention mechanism.
        attention_norm (nn.LayerNorm): Layer normalization applied after the attention mechanism.
        ffn (nn.Sequential): Feed-forward neural network.
        ffn_norm (nn.LayerNorm): Layer normalization applied after the feed-forward network.
        dropout1 (nn.Dropout): Dropout applied after the attention mechanism.
        dropout2 (nn.Dropout): Dropout applied after the feed-forward network.
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
        """
        Perform a forward pass through the encoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, embedding_dim) after applying
                          multi-head attention, dropout, layer normalization, and feed-forward network.
        """
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
    Encoder module for a Transformer model.
    
    Attributes:
        layers (nn.ModuleList): List of self-attention layers.
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Passes the input tensor through each self-attention layer in the encoder.
    """
    
    def __init__(self, num_layers: int = 1, d_model: int = 256, n_heads: int = 8):
        """
        Initializes the Encoder.

        Args:
            num_layers (int): The number of self-attention layers in the encoder. Default is 1.
            d_model (int): The dimension of the model. Default is 256.
            n_heads (int): The number of attention heads. Default is 8.
        """
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [SelfAttentionLayer(d_model, n_heads) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes the input tensor through each layer of the encoder.
        Args:
            x (torch.Tensor): The input tensor to be processed by the encoder layers.
        Returns:
            torch.Tensor: The output tensor after being processed by all encoder layers.
        """
        for layer in self.layers:
            x = layer(x)

        return x


