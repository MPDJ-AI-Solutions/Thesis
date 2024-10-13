from typing import Type

from torch import nn


class FFN(nn.Module):
    def __init__(self, d_model, dim_feedforward, activation: Type[nn.Module] = nn.ReLU, dropout = 0.1,):
        super(FFN, self).__init__()
        self.linear_layers = [nn.Linear(d_model, dim_feedforward), nn.Linear(dim_feedforward, d_model)]
        self.norm_layer =nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_layers =  [nn.Dropout(dropout), nn.Dropout(dropout)]
        self.activation = activation()


    def forward(self, x):
        x = self.norm_layer(x)
        x = self.linear_layers[0](x)
        x = self.activation(x)
        x = self.dropout_layers[0](x)
        x = self.linear_layers[1](x)
        x = self.dropout_layers[1](x)

        return x
