from abc import abstractmethod

import torch
import pandas as pd

class MeasureTool:
    """
    Abstract class for measure tool to measure performance metrics for both CNN and Transformer outputs on CUDA tensors.
    All abstract methods signatures are created dynamically in __init__.py file.
    """
    def compute_measures(self, result: torch.Tensor, target: torch.Tensor) -> pd.DataFrame:
        return pd.DataFrame({
            "TP"          : self.tp(result, target),
            "FP"          : self.fp(result, target),
            "FN"          : self.fn(result, target),
            "TN"          : self.tn(result, target),
            "Precision"   : self.precision(result, target),
            "Sensitivity" : self.sensitivity(result, target),
            "Specificity" : self.specificity(result, target),
            "NPV"         : self.npv(result, target),
            "FPR"         : self.fpr(result, target),
            "Accuracy"    : self.accuracy(result, target),
            "F-Score"     : self.fscore(result, target),
            "IoU"         : self.iou(result, target),
            "MCC"         : self.mcc(result, target),
            "AUC"         : self.auc(result, target),
            "CI"          : self.ci(result, result),
        })