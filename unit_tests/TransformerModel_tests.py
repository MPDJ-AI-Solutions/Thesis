import unittest

import torch
from numpy.testing import assert_equal

from models.TransformerMethaneDetection.model import TransformerModel as Model

class STARCOPDatasetTests(unittest.TestCase):
    def test_TransformerModel(self):
        bs, ch, w, h = 16, 8, 512, 512
        d_model = 2048
        model = Model(d_model=d_model).to("cuda")
        image = torch.rand(bs, ch, w, h).to("cuda")

        # Act
        for param in model.parameters():
            param.requires_grad = False
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            bbox, confidence, mask = model(image)

        # Assert
        assert_equal(mask.shape, (2, 1, 512, 512))
        assert_equal(confidence.shape, (2, 100, 1))
        assert_equal(bbox.shape, (2, 100, 4))

if __name__ == '__main__':
    unittest.main()
