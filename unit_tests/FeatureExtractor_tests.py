import unittest

import torch
from numpy.ma.testutils import assert_equal

from models.TransformerMethaneDetection.SpectralFeatureGenerator.feature_extractor import FeatureExtractor


class FeatureExtractorTests(unittest.TestCase):
    def test_FE(self):
        # Arrange
        # bs, c, h, w
        d_model = 256
        bs, d_model, h, w = 16, 256, 256, 256
        filtered_image = torch.rand( bs, 1, h, w)
        
        fe = FeatureExtractor(d_model=d_model)

        # Act
        result = fe(filtered_image)


        # Assert
        assert_equal(result.shape, (bs, d_model, int(h / 32), int(w/ 32)))


if __name__ == '__main__':
    unittest.main()
