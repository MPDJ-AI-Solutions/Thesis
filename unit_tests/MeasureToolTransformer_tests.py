import unittest
import torch

from sklearn.metrics import matthews_corrcoef

from models.Tools.measures.model_type import ModelType
from models.Tools.measures.measure_tool_factory import MeasureToolFactory


class MeasureToolTransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mt_transformer = MeasureToolFactory.get_measure_tool(ModelType.TRANSFORMER)
        # (4, 1, 3, 3)
        target_tensor = torch.Tensor([
            1.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 0.0,

            1.0, 1.0, 1.0,
            0.0, 0.0, 0.0,
            1.0, 1.0, 1.0,

            1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            1.0, 1.0, 0.0,

            0.0, 1.0, 1.0,
            1.0, 0.0, 1.0,
            0.0, 1.0, 0.0
        ])

        result_tensor = torch.Tensor([
            1.0, 0.0, 1.0,    # 1.0, 0.0, 1.0
            0.0, 0.0, 1.0,    # 0.0, 0.0, 1.0,
            1.0, 1.0, 1.0,    # 0.0, 0.0, 0.0, (3)

            1.0, 1.0, 1.0,    # 1.0, 1.0, 1.0,
            0.0, 0.0, 0.0,    # 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,    # 1.0, 1.0, 1.0, (2)

            1.0, 1.0, 1.0,    # 1.0, 0.0, 0.0, (2)
            0.0, 0.0, 0.0,    # 0.0, 0.0, 0.0,
            1.0, 1.0, 0.0,    # 1.0, 1.0, 0.0,

            0.0, 1.0, 1.0,    # 0.0, 1.0, 1.0,
            1.0, 0.0, 1.0,    # 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0     # 0.0, 1.0, 0.0  (3)
        ])

        # Change to valid shape
        cls.target = target_tensor.view(4, 1, 3, 3)
        cls.result = result_tensor.view(4, 1, 3, 3)

    # tp = 14, tn = 12, fn = 3, fp = 7
    def test_tp(self):
        # Arrange
        desired = 14 / 36

        # Act
        actual = self.mt_transformer.tp(self.result, self.target)

        # Assert
        self.assertAlmostEqual(first=desired, second=actual, places=4)


    def test_tn(self):
        # Arrange
        desired = 12 / 36

        # Act
        actual = self.mt_transformer.tn(self.result, self.target)

        # Assert
        self.assertAlmostEqual(first=desired, second=actual, places=4)


    def test_fn(self):
        # Arrange
        desired = 3 / 36

        # Act
        actual = self.mt_transformer.fn(self.result, self.target)

        # Assert
        self.assertAlmostEqual(first=desired, second=actual, places=4)


    def test_fp(self):
        # Arrange
        desired = 7 / 36

        # Act
        actual = self.mt_transformer.fp(self.result, self.target)

        # Assert
        self.assertAlmostEqual(first=desired, second=actual, places=4)


    def test_precision(self):
        # Arrange
        desired = 14 / (7 + 14 + 1e-6)

        # Act
        actual = self.mt_transformer.precision(target=self.target, result=self.result)

        # Assert
        self.assertAlmostEqual(first=actual, second=desired, places=4)


    def test_sensitivity(self):
        # Arrange
        desired = 14 / (3 + 14 + 1e-6)

        # Act
        actual = self.mt_transformer.sensitivity(target=self.target, result=self.result)

        # Assert
        self.assertAlmostEqual(first=actual, second=desired, places=4)

    def test_specificity(self):
        # Arrange
        desired = 12 / (7 + 12 + 1e-6)

        # Act
        actual = self.mt_transformer.specificity(target=self.target, result=self.result)

        # Assert
        self.assertAlmostEqual(first=actual, second=desired, places=4)

    def test_npv(self):
        # Arrange
        desired = 12 / (3 + 12 + 1e-6)

        # Act
        actual = self.mt_transformer.npv(target=self.target, result=self.result)

        # Assert
        self.assertAlmostEqual(first=actual, second=desired, places=4)

    def test_fpr(self):
        # Arrange
        desired = 7 / (7 + 12 + 1e-6)

        # Act
        actual = self.mt_transformer.fpr(target=self.target, result=self.result)

        # Assert
        self.assertAlmostEqual(first=actual, second=desired, places=4)


    def test_accuracy(self):
        # Arrange
        desired = (14 + 12) / (14 + 12 + 3 + 7 + 1e-6)

        # Act
        actual = self.mt_transformer.accuracy(target=self.target, result=self.result)

        # Assert
        self.assertAlmostEqual(first=actual, second=desired, places=4)


    def test_fscore(self):
        # Arrange
        precision = 14 / (7 + 14 + 1e-6)
        recall = 14 / (3 + 14 + 1e-6)
        desired = (2 * precision * recall) / (precision + recall + 1e-6)

        # Act
        actual = self.mt_transformer.fscore(target=self.target, result=self.result)

        # Assert
        self.assertAlmostEqual(first=actual, second=desired, places=4)

    # tp = 14, tn = 12, fn = 3, fp = 7
    def test_iou(self):
        # Arrange
        desired = 14 / (14 + 3 + 7 + 1e-6)

        # Act
        actual = self.mt_transformer.iou(target=self.target, result=self.result)

        # Assert
        self.assertAlmostEqual(first=actual, second=desired, places=4)


    def test_mcc(self):
        # Arrange
        desired = matthews_corrcoef(y_pred=self.result.flatten(), y_true=self.target.flatten())

        # Act
        actual = self.mt_transformer.mcc(target=self.target, result=self.result)

        # Assert
        self.assertAlmostEqual(first=actual, second=desired, places=4)

    #TODO
    def test_auc(self):
        # Arrange
        desired = 0

        # Act
        actual = self.mt_transformer.auc(target=self.target, result=self.result)

        # Assert
        self.assertAlmostEqual(first=actual, second=desired, places=4)

    #TODO
    def test_ci(self):
        # Arrange
        desired = 0

        # Act
        actual = self.mt_transformer.ci(target=self.target, result=self.result)

        # Assert
        self.assertAlmostEqual(first=actual, second=desired, places=4)


if __name__ == '__main__':
    unittest.main()