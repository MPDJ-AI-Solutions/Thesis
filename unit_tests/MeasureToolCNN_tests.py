import unittest
import torch

from numpy.ma.testutils import assert_equal
from measures.measure_tool_factory import MeasureToolFactory
from measures.model_type import ModelType


class MeasureToolCNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mt_cnn = MeasureToolFactory.get_measure_tool(ModelType.CNN)
        cls.target = torch.Tensor([1, 1, 0, 0, 0, 1, 1, 1])
        cls.result = torch.Tensor([1, 0, 0, 0, 1, 1, 1, 1])

    def test_tp(self):
        # Arrange
        desired = 0.5

        # Act
        actual = self.mt_cnn.tp(target=self.target, result=self.result)

        # Assert
        assert_equal(actual=actual, desired=desired)


    def test_tn(self):
        # Arrange
        desired = 0.25

        # Act
        actual = self.mt_cnn.tn(target=self.target, result=self.result)

        # Assert
        assert_equal(actual=actual, desired=desired)


    def test_fn(self):
        # Arrange
        desired = 0.125

        # Act
        actual = self.mt_cnn.fn(target=self.target, result=self.result)

        # Assert
        assert_equal(actual=actual, desired=desired)


    def test_fp(self):
        # Arrange
        desired = 0.125

        # Act
        actual = self.mt_cnn.fp(target=self.target, result=self.result)

        # Assert
        assert_equal(actual=actual, desired=desired)


    def test_precision(self):
        # Arrange
        desired = 0.8

        # Act
        actual = self.mt_cnn.precision(target=self.target, result=self.result)

        # Assert
        assert_equal(actual=actual, desired=desired)


    def test_sensitivity(self):
        # Arrange
        desired = 0.8

        # Act
        actual = self.mt_cnn.sensitivity(target=self.target, result=self.result)

        # Assert
        assert_equal(actual=actual, desired=desired)


    def test_specificity(self):
        # Arrange
        desired = 2/3

        # Act
        actual = self.mt_cnn.specificity(target=self.target, result=self.result)

        # Assert
        assert_equal(actual=actual, desired=desired)


    def test_accuracy(self):
        # Arrange
        desired = 0.75

        # Act
        actual = self.mt_cnn.accuracy(target=self.target, result=self.result)

        # Assert
        assert_equal(actual=actual, desired=desired)


    def test_fscore(self):
        # Arrange
        desired = 0.8

        # Act
        actual = self.mt_cnn.fscore(target=self.target, result=self.result)
        actual = round(actual, 1)

        # Assert
        assert_equal(actual=actual, desired=desired)


    def test_iou(self):
        # Arrange
        desired = 2/3

        # Act
        actual = self.mt_cnn.iou(target=self.target, result=self.result)

        # Assert
        assert_equal(actual=actual, desired=desired)


    def test_npv(self):
        # Arrange
        desired = 2/3

        # Act
        actual = self.mt_cnn.npv(target=self.target, result=self.result)

        # Assert
        assert_equal(actual=actual, desired=desired)


    def test_fpr(self):
        # Arrange
        desired = 1/3

        # Act
        actual = self.mt_cnn.fpr(target=self.target, result=self.result)

        # Assert
        assert_equal(actual=actual, desired=desired)


    def test_mcc(self):
        # Arrange
        desired = 0.4666666666666667

        # Act
        actual = self.mt_cnn.mcc(target=self.target, result=self.result)

        # Assert
        assert_equal(actual=actual, desired=desired)


    def test_auc(self):
        # Arrange
        desired = 0

        # Act
        actual = self.mt_cnn.auc(target=self.target, result=self.result)

        # Assert
        assert_equal(actual=0, desired=desired)


    def test_ci(self):
        # Arrange
        desired = 0

        # Act
        actual = self.mt_cnn.ci(target=self.target, result=self.result)

        # Assert
        assert_equal(actual=0, desired=desired)


if __name__ == '__main__':
    unittest.main()