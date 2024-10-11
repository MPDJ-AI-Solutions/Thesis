import unittest

from torch.utils.data import DataLoader

from models.TransformerMethaneDetection.Backbone.backbone import Backbone
from dataset.STARCOP_dataset import STARCOPDataset
from dataset.dataset_type import DatasetType
from dataset.dataset_info import transformer_input_converter


class BackboneTests(unittest.TestCase):
    def test_backbone_load_data(self):
        # Arrange
        dataset = STARCOPDataset(data_path=r"data", data_type=DatasetType.UNITTEST)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        model = Backbone(8)
        exception = False

        # Act
        try:
            for batch in dataloader:
                images, labels = transformer_input_converter(batch)
                model(images)
                break
        except Exception:
            exception = True

        # Assert
        self.assertEqual(exception, False)
