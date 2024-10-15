from typing import Optional

from torch import nn, Tensor

from models.TransformerMethaneDetection.Transformer.ffn import FFN


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout, activation = nn.ReLU):
        super(Decoder, self).__init__()
        self.decoder_layers = [DecoderLayer(d_model, nhead, dim_feedforward, dropout, activation) for _ in range(num_layers)]
        self.norm_layer = nn.LayerNorm(d_model)

    def forward(self, x,
                memory,
                mf_query,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,):
        intermediate = []
        for layer in self.decoder_layers:
            x = layer(x,
                      memory,
                      mf_query,
                      tgt_mask=tgt_mask,
                      memory_mask=memory_mask,
                      tgt_key_padding_mask=tgt_key_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask,
                      pos=pos,
                      query_pos=query_pos
            )

        return self.norm_layer(x).unsqueeze(0)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=nn.ReLU):
        super(DecoderLayer, self).__init__()
        self.fnn = FFN(d_model, dim_feedforward, dropout=dropout, activation=activation)
        self.norm_layers = [nn.LayerNorm(d_model, eps=1e-6), ]
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.dropout_layers = [nn.Dropout(dropout), nn.Dropout(dropout)]
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead)

    def forward(self, x, mask, position_embedding, query_position_embedding, key_padding_mask, memory, memory_mask, memory_key_padding_mask):
         tgt2 = self.norm_layers[0](x)
         q = k = tgt2 + query_position_embedding if query_position_embedding is not None else x
         tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=mask, key_padding_mask=key_padding_mask)[0]
         x = x + self.dropout[0](tgt2)
         tgt2 = self.norm_layers[1](x)
         tgt2 = self.multihead_attn(
             query= tgt2 + query_position_embedding if query_position_embedding is not None else x,
             key=self.with_pos_embed(memory, position_embedding),
             value=memory,
             attn_mask=memory_mask,
             key_padding_mask=memory_key_padding_mask,
         )[0]
         x = x + self.dropout[1](tgt2)
         x = x + self.fnn(x)
         return x

