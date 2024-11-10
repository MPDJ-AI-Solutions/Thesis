import unittest

import torch
from numpy.ma.testutils import assert_equal

from models.TransformerMethaneDetection.Transformer.query_refiner import QueryRefiner


class QueryRefinerTests(unittest.TestCase):
    def test_QR(self):
        # Arrange
        d_model = 2048
        num_queries = 100
        H, W = 16, 16  # Dimensions of fmc (height and width of feature map)
        bs = 2

        fmc = torch.randn(bs, H, W, d_model)  # Example feature map
        qr_module = QueryRefiner(d_model=d_model, num_queries=num_queries)

        # Act
        q_ref = qr_module(fmc)

        # Assert
        assert_equal(q_ref.shape, (2, 100, 2048))
        print("Refined Queries Shape:", q_ref.shape)  # Expected: (batch_size, num_queries, d_model)


if __name__ == '__main__':
    unittest.main()

