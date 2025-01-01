import unittest

import torch
from numpy.testing import assert_equal

from models.Transformer.MethaneMapper.Transformer.encoder import Encoder
from models.Transformer.MethaneMapper.Transformer.position_encoding import PositionalEncodingMM


class EncoderTests(unittest.TestCase):
    def test_Encoder(self):
        # Arrange
        H, W = 512, 512
        d_model = 256
        bs = 2

        encoder_input = torch.rand(bs, int(H/32) * int(W/32), d_model)

        model = Encoder(d_model=d_model, n_heads=8, num_layers=5)

        # Act
        methane_concentration_map = model(encoder_input)

        # Assert
        assert_equal(methane_concentration_map.shape, (bs, int((H/32) * (W/32)), d_model))


if __name__ == '__main__':
    unittest.main()