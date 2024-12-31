import unittest

from dataset.STARCOP_dataset import STARCOPDataset, DatasetType
from dataset.dataset_info import DatasetInfo

class STARCOPDatasetTests(unittest.TestCase):
    def test_STARCOPDataset_correct_path(self):
        # Arrange
        class TestDatasetInfo(DatasetInfo):
            @staticmethod
            def load_tensor(path: str, grid_id:int=0, crop_size:int=1):
                return path

        dataset = STARCOPDataset(data_path="data",
                                 data_type=DatasetType.UNITTEST,
                                 image_info_class=TestDatasetInfo)

        # Act
        path = dataset.__getitem__(0)

        # Assert
        self.assertEqual(path, r'data\STARCOP_ut\STARCOP_ut\ang20191018t141549_r6656_c0_w512_h512')


if __name__ == '__main__':
    unittest.main()
