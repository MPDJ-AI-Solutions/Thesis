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
        images_a, images_w, mag1c, labels = dataset.__getitem__(0)

        # Assert
        # Images AVIRIS
        self.assertEqual(len(images_a), 8)

        # Images WorldView3
        self.assertEqual(len(images_w), 8)

        # Images mag1c
        self.assertIn("weight", mag1c.keys())
        self.assertIn("mag1c", mag1c.keys())

        # Labels
        self.assertIn("label_rgba", labels.keys())
        self.assertIn("label_binary", labels.keys())
        self.assertIn("label_string", labels.keys())


if __name__ == '__main__':
    unittest.main()
