import unittest

from models.TransformerMethaneDetection.Segmentation.segmentation import BoxAndMaskPredictor

class STARCOPDatasetTests(unittest.TestCase):
    def test_TransformerModel(self):
        # Arrange
        BaM_predictor = BoxAndMaskPredictor()
        
        # Act

        # Assert

if __name__ == '__main__':
    unittest.main()
