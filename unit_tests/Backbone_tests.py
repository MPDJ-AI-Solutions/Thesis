import unittest
import torch

from numpy.ma.testutils import assert_equal
from models.Transformer.MethaneMapper.Backbone.backbone import Backbone

class BackboneTests(unittest.TestCase):
    def test_backbone_load_data(self):
        # Arrange
        bs, ch, h, w = 16, 8, 256, 256
        out_channels = 2048
        d_model = 256
        hsi = torch.rand(bs, ch, h, w)
        
        backbone = Backbone(rgb_channels=3, swir_channels=5, out_channels=out_channels)
        
        # Act
        result = backbone(hsi)

        # Assert
        assert_equal(result.shape, (bs, d_model, int(h / 32), int(w / 32)))

if __name__ == '__main__':
    unittest.main()
