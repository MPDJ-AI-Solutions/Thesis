import torch
import math
from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score

from .measure_tool import MeasureTool


class MeasureToolTransformer(MeasureTool):
    @staticmethod
    def tp(result: torch.Tensor, target: torch.Tensor) -> float:
        return ((result == 1) & (target == 1)).float().mean().item()


    @staticmethod
    def fp(result: torch.Tensor, target: torch.Tensor) -> float:
        return ((result == 1) & (target == 0)).float().mean().item()


    @staticmethod
    def fn(result: torch.Tensor, target: torch.Tensor) -> float:
        return ((result == 0) & (target == 1)).float().mean().item()


    @staticmethod
    def tn(result: torch.Tensor, target: torch.Tensor) -> float:
        return ((result == 0) & (target == 0)).float().mean().item()


    @staticmethod
    def precision(result: torch.Tensor, target: torch.Tensor) -> float:
        tp = MeasureToolTransformer.tp(result, target)
        fp = MeasureToolTransformer.fp(result, target)
        return tp / (tp + fp + 1e-6)


    @staticmethod
    def sensitivity(result: torch.Tensor, target: torch.Tensor) -> float:
        tp = MeasureToolTransformer.tp(result, target)
        fn = MeasureToolTransformer.fn(result, target)
        return tp / (tp + fn + 1e-6)


    @staticmethod
    def specificity(result: torch.Tensor, target: torch.Tensor) -> float:
        tn = MeasureToolTransformer.tn(result, target)
        fp = MeasureToolTransformer.fp(result, target)
        return tn / (tn + fp + 1e-6)


    @staticmethod
    def npv(result: torch.Tensor, target: torch.Tensor) -> float:
        tn = MeasureToolTransformer.tn(result, target)
        fn = MeasureToolTransformer.fn(result, target)
        return tn / (tn + fn + 1e-6)


    @staticmethod
    def fpr(result: torch.Tensor, target: torch.Tensor) -> float:
        fp = MeasureToolTransformer.fp(result, target)
        tn = MeasureToolTransformer.tn(result, target)
        return fp / (fp + tn + 1e-6)


    @staticmethod
    def accuracy(result: torch.Tensor, target: torch.Tensor) -> float:
        tp = MeasureToolTransformer.tp(result, target)
        fp = MeasureToolTransformer.fp(result, target)
        fn = MeasureToolTransformer.fn(result, target)
        tn = MeasureToolTransformer.tn(result, target)
        numerator = tp + tn
        denominator = tp + fp + fn + tn
        return numerator / (denominator + 1e-6)


    @staticmethod
    def fscore(result: torch.Tensor, target: torch.Tensor) -> float:
        precision = MeasureToolTransformer.precision(result, target)
        recall = MeasureToolTransformer.sensitivity(result, target)
        return 2 * (precision * recall) / (precision + recall + 1e-6)


    @staticmethod
    def iou(result: torch.Tensor, target: torch.Tensor) -> float:
        intersection = ((result == 1) & (target == 1)).float().sum().item()
        union = ((result == 1) | (target == 1)).float().sum().item()
        return intersection / (union + 1e-6)


    @staticmethod
    def mcc(result: torch.Tensor, target: torch.Tensor) -> float:
        tp = MeasureToolTransformer.tp(result, target)
        fp = MeasureToolTransformer.fp(result, target)
        fn = MeasureToolTransformer.fn(result, target)
        tn = MeasureToolTransformer.tn(result, target)
        numerator = (tp * tn - fp * fn)
        denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return numerator / (denominator + 1e-6)


    @staticmethod
    def auc(result: torch.Tensor, target: torch.Tensor) -> float:
        return roc_auc_score(target.flatten(), result.flatten())


    @staticmethod
    def ci(result: torch.Tensor, target: torch.Tensor) -> float:
        std = result.std().item()
        ci = std * 1.96 / (result.numel() ** 0.5)
        return ci