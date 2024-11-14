import torch
from abc import abstractmethod
from .measure_tool import MeasureTool


class MethodLoader:
    """
    Class which purpose is to load static, abstract methods to MeasureTool class.
    """
    @staticmethod
    @abstractmethod
    def new_abstract_method(result: torch.Tensor, target: torch.Tensor):
        pass


    @staticmethod
    def load_methods(methods):
        for method in methods:
            setattr(MeasureTool, method, MethodLoader.new_abstract_method)


method_names = [
    "tp", "fp", "fn", "tn", "precision", "sensitivity", "specificity",
    "npv", "fpr", "accuracy", "fscore", "iou", "mcc", "auc", "ci"
]


MethodLoader.load_methods(method_names)