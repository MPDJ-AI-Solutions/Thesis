import torch
import math
from sklearn.metrics import roc_auc_score

from .measure_tool import MeasureTool


class MeasureToolTransformer(MeasureTool):
    """
    MeasureToolTransformer is a class that provides various static methods to compute 
    different performance metrics for binary classification tasks using PyTorch tensors.
    """

    @staticmethod
    def tp(result: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate the true positive rate.

        Args:
            result (torch.Tensor): The predicted tensor with binary values (0 or 1).
            target (torch.Tensor): The ground truth tensor with binary values (0 or 1).

        Returns:
            float: The mean of true positives.
        """
        return ((result == 1) & (target == 1)).float().mean().item()


    @staticmethod
    def fp(result: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate the false positive rate.

        Args:
            result (torch.Tensor): The predicted tensor, where each element is a prediction (1 for positive, 0 for negative).
            target (torch.Tensor): The ground truth tensor, where each element is the true label (1 for positive, 0 for negative).

        Returns:
            float: The mean false positive rate.
        """
        return ((result == 1) & (target == 0)).float().mean().item()


    @staticmethod
    def fn(result: torch.Tensor, target: torch.Tensor) -> float:
        """
        Computes the mean of false negatives in the prediction results.

        Args:
            result (torch.Tensor): The predicted tensor.
            target (torch.Tensor): The ground truth tensor.

        Returns:
            float: The mean of false negatives, where a false negative is defined as
                   a case where the prediction is 0 and the ground truth is 1.
        """
        return ((result == 0) & (target == 1)).float().mean().item()


    @staticmethod
    def tn(result: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate the true negative rate.

        Args:
            result (torch.Tensor): The predicted tensor.
            target (torch.Tensor): The ground truth tensor.

        Returns:
            float: The mean of true negatives in the result tensor.
        """
        return ((result == 0) & (target == 0)).float().mean().item()


    @staticmethod
    def precision(result: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate the precision metric for the given result and target tensors.

        Precision is defined as the ratio of true positives (tp) to the sum of true positives and false positives (fp).
        It is a measure of the accuracy of the positive predictions.

        Args:
            result (torch.Tensor): The predicted tensor.
            target (torch.Tensor): The ground truth tensor.

        Returns:
            float: The precision value.
        """
        tp = MeasureToolTransformer.tp(result, target)
        fp = MeasureToolTransformer.fp(result, target)
        return tp / (tp + fp + 1e-6)


    @staticmethod
    def sensitivity(result: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate the sensitivity (true positive rate) of the given results.

        Sensitivity is defined as the ratio of true positives (tp) to the sum of true positives and false negatives (fn).

        Args:
            result (torch.Tensor): The predicted results.
            target (torch.Tensor): The ground truth labels.

        Returns:
            float: The sensitivity value.
        """
        tp = MeasureToolTransformer.tp(result, target)
        fn = MeasureToolTransformer.fn(result, target)
        return tp / (tp + fn + 1e-6)


    @staticmethod
    def specificity(result: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate the specificity metric.

        Specificity, also known as the true negative rate, measures the proportion of actual negatives that are correctly identified as such.

        Args:
            result (torch.Tensor): The predicted values.
            target (torch.Tensor): The ground truth values.

        Returns:
            float: The specificity value.
        """
        tn = MeasureToolTransformer.tn(result, target)
        fp = MeasureToolTransformer.fp(result, target)
        return tn / (tn + fp + 1e-6)


    @staticmethod
    def npv(result: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate the Negative Predictive Value (NPV) for the given result and target tensors.

        NPV is defined as the ratio of true negatives (TN) to the sum of true negatives (TN) and false negatives (FN).

        Args:
            result (torch.Tensor): The predicted values tensor.
            target (torch.Tensor): The ground truth values tensor.

        Returns:
            float: The calculated NPV value.
        """
        tn = MeasureToolTransformer.tn(result, target)
        fn = MeasureToolTransformer.fn(result, target)
        return tn / (tn + fn + 1e-6)


    @staticmethod
    def fpr(result: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate the False Positive Rate (FPR).

        The FPR is calculated as the number of false positives divided by the 
        sum of false positives and true negatives.

        Args:
            result (torch.Tensor): The predicted results.
            target (torch.Tensor): The ground truth labels.

        Returns:
            float: The false positive rate.
        """
        fp = MeasureToolTransformer.fp(result, target)
        tn = MeasureToolTransformer.tn(result, target)
        return fp / (fp + tn + 1e-6)


    @staticmethod
    def accuracy(result: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate the accuracy of the predictions.

        Accuracy is defined as the ratio of correctly predicted samples (true positives and true negatives)
        to the total number of samples.

        Args:
            result (torch.Tensor): The predicted labels.
            target (torch.Tensor): The true labels.

        Returns:
            float: The accuracy of the predictions.
        """
        tp = MeasureToolTransformer.tp(result, target)
        fp = MeasureToolTransformer.fp(result, target)
        fn = MeasureToolTransformer.fn(result, target)
        tn = MeasureToolTransformer.tn(result, target)
        numerator = tp + tn
        denominator = tp + fp + fn + tn
        return numerator / (denominator + 1e-6)


    @staticmethod
    def fscore(result: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate the F1 score given the result and target tensors.

        The F1 score is the harmonic mean of precision and recall, providing a balance
        between the two metrics. It is particularly useful for imbalanced datasets.

        Args:
            result (torch.Tensor): The predicted values.
            target (torch.Tensor): The ground truth values.

        Returns:
            float: The F1 score.
        """
        precision = MeasureToolTransformer.precision(result, target)
        recall = MeasureToolTransformer.sensitivity(result, target)
        return 2 * (precision * recall) / (precision + recall + 1e-6)


    @staticmethod
    def iou(result: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate the Intersection over Union (IoU) between two tensors.

        Args:
            result (torch.Tensor): The predicted tensor with binary values (0 or 1).
            target (torch.Tensor): The ground truth tensor with binary values (0 or 1).

        Returns:
            float: The IoU score, a value between 0 and 1.
        """
        intersection = ((result == 1) & (target == 1)).float().sum().item()
        union = ((result == 1) | (target == 1)).float().sum().item()
        return intersection / (union + 1e-6)


    @staticmethod
    def mcc(result: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate the Matthews correlation coefficient (MCC) for the given result and target tensors.

        The MCC is a measure of the quality of binary classifications, taking into account true and false positives and negatives.
        It returns a value between -1 and 1, where 1 indicates perfect prediction, 0 indicates random prediction, and -1 indicates
        inverse prediction.

        Args:
            result (torch.Tensor): The predicted values tensor.
            target (torch.Tensor): The ground truth values tensor.

        Returns:
            float: The calculated MCC value.
        """
        tp = MeasureToolTransformer.tp(result, target)
        fp = MeasureToolTransformer.fp(result, target)
        fn = MeasureToolTransformer.fn(result, target)
        tn = MeasureToolTransformer.tn(result, target)
        numerator = (tp * tn - fp * fn)
        denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return numerator / (denominator + 1e-6)


    @staticmethod
    def auc(result: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute the Area Under the Receiver Operating Characteristic Curve (ROC AUC) 
        from prediction scores.

        Args:
            result (torch.Tensor): The prediction scores.
            target (torch.Tensor): The ground truth binary labels.

        Returns:
            float: The computed ROC AUC score.
        """
        return roc_auc_score(target.flatten(), result.flatten())

    @staticmethod
    def ci(result: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate the confidence interval (CI) for the given result tensor.

        Args:
            result (torch.Tensor): The tensor containing the results.
            target (torch.Tensor): The tensor containing the target values (not used in the calculation).

        Returns:
            float: The calculated confidence interval.
        """
        std = result.float().std().item()
        ci = std * 1.96 / (result.float().numel() ** 0.5)
        return ci