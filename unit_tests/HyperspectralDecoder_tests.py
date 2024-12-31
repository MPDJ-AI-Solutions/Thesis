import unittest

import torch
from numpy.ma.testutils import assert_equal

from models.Transformer.MethaneMapper.Transformer.encoder import Encoder
from models.Transformer.MethaneMapper.Transformer.position_encoding import PositionalEncodingMM
from models.Transformer.MethaneMapper.Transformer.query_refiner import QueryRefiner
from models.Transformer.MethaneMapper.Transformer.hyperspectral_decoder import HyperspectralDecoder


class HyperspectralDecoderTests(unittest.TestCase):
    def test_HD(self):
        # Arrange
        H, W, bs, d_model = 256, 256, 8, 256  # Example dimensions
        n_heads = 8
        num_layers = 6
        num_queries = 100

        fmc = torch.randn(bs, H, W, d_model)  # Example feature map
        qr_module = QueryRefiner(d_model=d_model, num_queries=num_queries)
        q_ref = qr_module(fmc)

        hsi = torch.rand(bs, int(H/32), int(W/32), d_model)
        pos_encoder = PositionalEncodingMM(d_model=d_model)
        encoder = Encoder(d_model=d_model, n_heads=n_heads, num_layers=5)

        encoder_output = encoder(pos_encoder(hsi)[0].flatten(2).permute(0, 2, 1))
        decoder = HyperspectralDecoder(d_model, n_heads, num_layers)

        # Act
        output_embeddings = decoder(pos_encoder(encoder_output.view(bs, int(H/32), int(W/32), d_model)), q_ref)

        # Assert
        assert_equal(output_embeddings.shape, (bs, num_queries, d_model))


if __name__ == '__main__':
    unittest.main()