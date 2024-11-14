from abc import abstractmethod

import torch
import pandas as pd

class MeasureTool:
    """
    Abstract class for measure tool to measure performance metrics for both CNN and Transformer outputs on CUDA tensors.
    """
    @staticmethod
    @abstractmethod
    def tp(result: torch.Tensor, target: torch.Tensor) -> float:
        pass


    @staticmethod
    @abstractmethod
    def fp(result: torch.Tensor, target: torch.Tensor) -> float:
        pass


    @staticmethod
    @abstractmethod
    def fn(result: torch.Tensor, target: torch.Tensor) -> float:
        pass


    @staticmethod
    @abstractmethod
    def tn(result: torch.Tensor, target: torch.Tensor) -> float:
        pass


    @staticmethod
    @abstractmethod
    def precision(result: torch.Tensor, target: torch.Tensor) -> float:
        pass


    @staticmethod
    @abstractmethod
    def sensitivity(result: torch.Tensor, target: torch.Tensor) -> float:
        pass


    @staticmethod
    @abstractmethod
    def specificity(result: torch.Tensor, target: torch.Tensor) -> float:
        pass


    @staticmethod
    @abstractmethod
    def accuracy(result: torch.Tensor, target: torch.Tensor) -> float:
        pass


    @staticmethod
    @abstractmethod
    def fscore(result: torch.Tensor, target: torch.Tensor) -> float:
        pass


    @staticmethod
    @abstractmethod
    def iou(result: torch.Tensor, target: torch.Tensor) -> float:
        pass


    @staticmethod
    @abstractmethod
    def npv(result: torch.Tensor, target: torch.Tensor) -> float:
        pass


    @staticmethod
    @abstractmethod
    def fpr(result: torch.Tensor, target: torch.Tensor) -> float:
        pass


    @staticmethod
    @abstractmethod
    def mcc(result: torch.Tensor, target: torch.Tensor) -> float:
        pass


    @staticmethod
    @abstractmethod
    def auc(result: torch.Tensor, target: torch.Tensor) -> float:
        pass


    @staticmethod
    @abstractmethod
    def ci(result: torch.Tensor, target: torch.Tensor) -> float:
        pass


    def compute_measures(self, result: torch.Tensor, target: torch.Tensor) -> pd.DataFrame:
        return pd.DataFrame({
            "TP"          : self.tp(result, target),
            "FP"          : self.fp(result, target),
            "FN"          : self.fn(result, target),
            "TN"          : self.tn(result, target),
            "PRECISION"   : self.precision(result, target),
            "Sensitivity" : self.sensitivity(result, target),
            "Specificity" : self.specificity(result, target),
            "Accuracy"    : self.accuracy(result, target),
            "F-Score"     : self.fscore(result, target),
            "IoU"         : self.iou(result, target),
            "NPV"         : self.npv(result, target),
            "FPR"         : self.fpr(result, target),
            "MCC"         : self.mcc(result, target),
            "AUC"         : self.auc(result, target),
            "CI"          : self.ci(result, result),
        })