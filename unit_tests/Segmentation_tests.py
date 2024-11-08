import unittest
import torch

from numpy.ma.testutils import assert_equal

from models.TransformerMethaneDetection.Segmentation.segmentation import BoxAndMaskPredictor

class STARCOPDatasetTests(unittest.TestCase):
    def test_TransformerModel(self):
        # Arrange
        bs, num_queries, d_model, H, W = 16, 100, 1024, 256, 256
        decoder_out = torch.rand(bs, num_queries, d_model)
        encoder_out = torch.rand(bs, int((H/32) * (W/32)), d_model)

        BaM_predictor = BoxAndMaskPredictor(embedding_dim=d_model, fpn_channels=100)
        
        # Act
        bbox, confidence, final_mask = BaM_predictor(decoder_out, encoder_out)
        
        # Assert
        print("BBox_prediction shape: ", bbox.shape)  
        print("confidence shape:", confidence.shape)  
        print("final mask shape:", final_mask.shape)  
        

if __name__ == '__main__':
    unittest.main()
