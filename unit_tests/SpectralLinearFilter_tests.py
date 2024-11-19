import unittest

import torch
from numpy.ma.testutils import assert_equal

from models.TransformerMethaneDetection.SpectralFeatureGenerator.spectral_linear_filter import SpectralLinearFilter, SpectralLinearFilterParallel


class SpectralLinearFilterTests(unittest.TestCase):
    def test_SLF(self):
        # Arrange
        # bs, c, w, h
        image = torch.rand(16, 8, 256, 256)
        methane_pattern = [0, 0, 0, 0.1, 0.3, 0.6, 0.8, 0.7]

        model = SpectralLinearFilter()

        # Act
        methane_concentration_map = model(image, methane_pattern)


        # Assert
        assert_equal(methane_concentration_map.shape, (16, 256, 256))
        print(methane_concentration_map.shape)

    def test_slf_parallel(self):
        # Arrange
        # bs, c, w, h
        for i in range(20):
            print(f'Running test {i}' )
            image = torch.rand(16, 8, 50, 50)
            methane_pattern = [0, 0, 0, 0.1, 0.3, 0.6, 0.8, 0.7]

            model_linear = SpectralLinearFilter()
            model_parallel = SpectralLinearFilterParallel()
            expected_methane_concentration_map = model_linear(image, methane_pattern)

            # Act
            result = model_parallel(image, methane_pattern)

            # Assert
            assert torch.equal(expected_methane_concentration_map, result)

if __name__ == '__main__':
    unittest.main()

