from torch import nn
from torch.nn.functional import dropout


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model=256, n_head=8, dropout=0.1):
        super(SelfAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.attention_norm = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn, _ = self.multihead_attn(x, x, x)
        x = self.attention_norm(x + self.dropout(attn))

        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + self.dropout(ffn_out))

        return x


class Encoder(nn.Module):
    def __init__(self, num_layers=1, d_model=256, n_heads=8):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [SelfAttentionLayer(d_model, n_heads) for _ in range(num_layers)]
        )

    def forward(self, x, pos_embedding):
        x = x + pos_embedding
        for layer in self.layers:
            x = layer(x)

        return x


