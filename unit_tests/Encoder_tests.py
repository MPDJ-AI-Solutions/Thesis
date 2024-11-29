import unittest

import torch
from numpy.testing import assert_equal

from models.Transformer.MethaneMapper import Encoder
from models.Transformer.MethaneMapper.Transformer.position_encoding import PositionalEncodingMM


class EncoderTests(unittest.TestCase):
    def test_Encoder(self):
        # Arrange
        H, W = 256, 256
        d_model = 256
        bs = 8
        backbone_output = torch.rand(bs, int(H/32), int(W/32), d_model)
        pos_encoder = PositionalEncodingMM(d_model=d_model, height=int(H / 32), width=int(W / 32))

        encoder_input = pos_encoder(backbone_output)

        model = Encoder(d_model=d_model, n_heads=8, num_layers=5)

        # Act
        methane_concentration_map = model(encoder_input)


        # Assert
        assert_equal(methane_concentration_map.shape, (bs, int((H/32) * (W/32)), d_model))
        print(methane_concentration_map.shape)


if __name__ == '__main__':
    unittest.main()