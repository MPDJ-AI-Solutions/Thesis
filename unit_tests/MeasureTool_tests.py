import unittest
import torch

from numpy.ma.testutils import assert_equal
from measures.measure_tool_cnn import MeasureToolCNN


class MeasureToolTests(unittest.TestCase):
    def test_tp_cnn(self):
        # Arrange
        target = torch.Tensor([1, 1, 1, 1])
        result = torch.Tensor([1, 0, 1, 0])
        desired = 0.5

        # Act
        actual = MeasureToolCNN.tp(target=target, result=result)

        # Assert
        assert_equal(actual=actual, desired=desired)


    def test_tn_cnn(self):
       pass


    def test_fn_cnn(self):
       pass


    def test_fp_cnn(self):
       pass


    def template_for_test(self):
        # Arrange

        # Act

        # Assert

        pass

if __name__ == '__main__':
    unittest.main()