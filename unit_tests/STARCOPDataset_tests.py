import unittest

from dataset.STARCOP_dataset import STARCOPDataset, DatasetType


class MyTestCase(unittest.TestCase):
    def test_STARCOPDataset_correct_length(self):
        # Arrange
        dataset = STARCOPDataset(data_path=r"data",
                                 data_type=DatasetType.UNITTEST)
        expected_length = 3

        # Act
        length = dataset.__len__()

        # Assert
        self.assertEqual(length, expected_length)

    def test_STARCOPDataset_get_item(self):
        # Arrange
        dataset = STARCOPDataset(data_path=r"data",
                                 data_type=DatasetType.UNITTEST)

        # Act
        dataset_info = dataset.__getitem__(0)

        # Assert
        # Images AVIRIS
        self.assertEqual(len(dataset_info.images_AVIRIS), 8)

        # Images WorldView3
        self.assertEqual(len(dataset_info.images_WV3), 8)

        # Images mag1c
        self.assertIn("weight", dataset_info.mag1c.keys())
        self.assertIn("mag1c", dataset_info.mag1c.keys())

        # Labels
        self.assertIn("label_rgba", dataset_info.labels.keys())
        self.assertIn("label_binary", dataset_info.labels.keys())
        self.assertIn("label_string", dataset_info.labels.keys())


if __name__ == '__main__':
    unittest.main()
