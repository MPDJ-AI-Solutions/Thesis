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

        encoder_output = torch.rand(bs, int(H/32) * int(W/32), d_model)
        q_ref_output = torch.rand(bs, num_queries, d_model)
        decoder = HyperspectralDecoder(d_model, n_heads, num_layers)

        # Act
        output_embeddings = decoder(encoder_output, q_ref_output)

        # Assert
        assert_equal(output_embeddings.shape, (bs, num_queries, d_model))


if __name__ == '__main__':
    unittest.main()