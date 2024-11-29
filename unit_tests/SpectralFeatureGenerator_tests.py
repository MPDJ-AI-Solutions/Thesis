import unittest
import torch

from numpy.ma.testutils import assert_equal
from models.Transformer.MethaneMapper.SpectralFeatureGenerator.spectral_feature_generator import SpectralFeatureGenerator

class SpectralFeatureGeneratorTests(unittest.TestCase):
    def test_SFG(self):
        # Arrange
        # bs, c, h, w
        d_model = 2048
        bs, ch, h, w = 16, 1, 512, 512
        
        input_image = torch.rand(bs, ch, h, w)
        sfg = SpectralFeatureGenerator(d_model=d_model)

        # Act
        result = sfg(input_image)

        # Assert
        assert_equal(result.shape, (bs, d_model, int(h / 32), int(w/ 32)))


if __name__ == '__main__':
    unittest.main()
