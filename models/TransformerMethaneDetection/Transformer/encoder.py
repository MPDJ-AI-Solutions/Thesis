from torch import nn

from .ffn import FFN


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead):
        super(Encoder, self).__init__()
        self.encoder_layers = [EncoderLayer(d_model, nhead) for _ in range(num_layers)]
        self.norm_layer = nn.LayerNorm(d_model)

    def forward(self, x, mask, positional_encoding):
        for layer in self.encoder_layers:
            x = layer(x, mask, positional_encoding)

        return self.norm_layer(x)



class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.ffn = FFN(d_model, dim_feedforward)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask, positional_encoding):
        x = self.layer_norm(x)
        q = k = x + positional_encoding if positional_encoding is not None else x
        src = self.self_attn(q, q, key_padding_mask=mask)
        x = x + self.dropout(src)
        x = x + self.ffn(x)

        return x
