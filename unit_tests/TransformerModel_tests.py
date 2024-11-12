import unittest

import torch
from numpy.testing import assert_equal
import torch.optim as optim
from torch.utils.data import DataLoader

from models.TransformerMethaneDetection.model import TransformerModel as Model

class STARCOPDatasetTests(unittest.TestCase):
    def test_TransformerModel(self):
        bs, ch, w, h = 5, 8, 512, 512
        d_model = 256
        model = Model(d_model=d_model).to("cuda")
        image = torch.rand(bs, ch, w, h).to("cuda")
        filtered_image = torch.rand(bs, 1, w, h).to("cuda")

        # Act
        bbox, confidence, mask = model(image, filtered_image)

        # Assert
        assert_equal(mask.shape, (bs, 1, h, w))
        assert_equal(confidence.shape, (bs, 100, 1))
        assert_equal(bbox.shape, (bs, 100, 4))


if __name__ == '__main__':
    unittest.main()
