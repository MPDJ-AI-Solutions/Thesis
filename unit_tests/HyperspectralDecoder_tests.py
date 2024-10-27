import unittest

import torch
from numpy.ma.testutils import assert_equal

from models.TransformerMethaneDetection.Transformer.query_refiner import QueryRefiner
from models.TransformerMethaneDetection.Transformer.hyperspectral_decoder import HyperspectralDecoder


class HyperspectralDecoder_tests(unittest.TestCase):
    def test_QR(self):
        # Example usage
        H, W, d_model = 32, 32, 256  # Example dimensions
        n_heads = 8
        num_layers = 6
        num_queries = 100

        # Init queries
        fmc = torch.randn(H, W, d_model)  # Example feature map
        qr_module = QueryRefiner(d_model=d_model, num_queries=num_queries)
        q_ref = qr_module(fmc)

        decoder = HyperspectralDecoder(d_model, n_heads, num_layers)
        encoder_output = torch.randn(H * W, 1, d_model)  # Shape: (H*W, batch_size, d_model)
        pos_embeddings = torch.randn(H * W, d_model)  # Shape: (H*W, d_model)
        pos_embeddings = pos_embeddings.unsqueeze(1)  # Shape: (H*W, 1, d_model)

        # Generate output embeddings
        output_embeddings = decoder(encoder_output, pos_embeddings, q_ref)

        # Assert
        assert_equal(output_embeddings.shape, (100, 1, 256))
        print("Decoder output:", output_embeddings.shape)  # Expected: (num_queries, d_model)


if __name__ == '__main__':
    unittest.main()