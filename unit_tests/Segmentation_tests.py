import unittest
import torch

from numpy.ma.testutils import assert_equal

from models.Transformer.MethaneMapper.Segmentation.segmentation import BoxAndMaskPredictor


class STARCOPDatasetTests(unittest.TestCase):
    def test_TransformerModel(self):
        # Arrange
        bs, num_queries, d_model, H, W = 16, 100, 2048, 512, 512
        decoder_out = torch.rand(bs, num_queries, d_model)
        encoder_out = torch.rand(bs, int((H/32) * (W/32)), d_model)

        output_predictor = BoxAndMaskPredictor(
            embedding_dim=d_model, fpn_channels=num_queries, result_height=H, result_width=W
        )

        # Act
        bbox, confidence, final_mask = output_predictor(decoder_out, encoder_out)

        # Assert
        assert_equal(bbox.shape, (bs, num_queries, 4))
        assert_equal(confidence.shape, (bs, num_queries, 1))
        assert_equal(final_mask.shape, (bs, 1, H, W))

        print("BBox_prediction shape: ", bbox.shape)
        print("confidence shape:", confidence.shape)
        print("final mask shape:", final_mask.shape)

if __name__ == '__main__':
    unittest.main()
