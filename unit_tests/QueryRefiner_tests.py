import unittest

import torch
from numpy.ma.testutils import assert_equal

from models.TransformerMethaneDetection.Transformer.query_refiner import QueryRefiner


class QueryRefiner_tests(unittest.TestCase):
    def test_QR(self):
        # Arrange
        d_model = 256
        num_queries = 100
        H, W = 256, 256  # Dimensions of fmc (height and width of feature map)

        fmc = torch.randn(H, W, d_model)  # Example feature map
        qr_module = QueryRefiner(d_model=d_model, num_queries=num_queries)

        # Act
        q_ref = qr_module(fmc)

        # Assert
        assert_equal(q_ref.shape, (100, 1, 256))
        print("Refined Queries Shape:", q_ref.shape)  # Expected: (num_queries, d_model)


if __name__ == '__main__':
    unittest.main()

