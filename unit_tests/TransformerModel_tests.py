import unittest

import torch
from numpy.testing import assert_equal

from models.Transformer.MethaneMapper.model import TransformerModel as Model

class STARCOPDatasetTests(unittest.TestCase):
    def test_TransformerModel(self):
        bs, ch, w, h = 5, 9, 512, 512
        d_model = 512
        model = Model(d_model=d_model)
        image = torch.rand(bs, ch, w, h)
        filtered_image = torch.rand(bs, 1, w, h)

        # Act
        logits = model(image, filtered_image)

        # Assert
        #assert_equal(mask.shape, (bs, 1, h, w))
        assert_equal(logits.shape, (bs, 2))

if __name__ == '__main__':
    unittest.main()
