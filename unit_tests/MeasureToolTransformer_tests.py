import unittest
import torch

from numpy.ma.testutils import assert_equal
from measures.measure_tool_factory import MeasureToolFactory
from measures.model_type import ModelType


class MeasureToolTransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mt_cnn = MeasureToolFactory.get_measure_tool(ModelType.TRANSFORMER)


    def test_tp(self):
        pass


    def test_tn(self):
        pass


    def test_fn(self):
        pass


    def test_fp(self):
        pass


    def template_for_test(self):
        # Arrange

        # Act

        # Assert

        pass

if __name__ == '__main__':
    unittest.main()