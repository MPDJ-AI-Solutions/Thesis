import torch
from torch import nn

from models.TransformerMethaneDetection.Transformer.decoder import Decoder
from models.TransformerMethaneDetection.Transformer.encoder import Encoder


class Transformer(nn.Module):
    def __init__(self, d_model, nhead, encoder_num_layers, num_decoder_layers, dim_feedforward=2048, dropout=0.1):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.encoder = Encoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=encoder_num_layers
        )

        self.decoder = Decoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_layers=num_decoder_layers
        )

    def forward(self, x, mask, pos_embed, query_embed, mf_query):
        bs, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        positional_embeddings = pos_embed[1].permute(2, 0, 1)
        query_embeddings = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mf_query = mf_query.permute(1, 0, 2)
        mask = mask.flatten(1)

        y = torch.zeros_like(query_embed)
        memory = self.encoder(
            x,
            key_padding_mask=mask,
            positional_embedings=positional_embeddings,
            query_embeddings=query_embeddings
        )
        hs = self.decoder(
            y,
            memory,
            mask,
            mf_query=mf_query,
            memory_key_padding_mask=mask,
            positional_embeddings=positional_embeddings,
            query_embeddings=query_embeddings
        )

        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
