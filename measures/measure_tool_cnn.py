import torch
import math

from .measure_tool import MeasureTool
from sklearn.metrics import roc_auc_score

class MeasureToolCNN(MeasureTool):
    """
    MeasureToolCNN is a class that provides various static methods to calculate different performance metrics for a CNN model.
    """

    @staticmethod
    def tp(result: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate the true positive rate (TPR) for binary classification.

        Args:
            result (torch.Tensor): The predicted labels tensor, where each element is either 0 or 1.
            target (torch.Tensor): The ground truth labels tensor, where each element is either 0 or 1.

        Returns:
            float: The true positive rate, which is the ratio of true positives to the total number of predictions.
        """
        return ((result == 1) & (target == 1)).float().sum().item() / result.size(0)


    @staticmethod
    def fp(result: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate the false positive rate.

        Args:
            result (torch.Tensor): The predicted tensor with binary values (0 or 1).
            target (torch.Tensor): The ground truth tensor with binary values (0 or 1).

        Returns:
            float: The false positive rate, which is the number of false positives 
                   divided by the total number of predictions.
        """
        return ((result == 1) & (target == 0)).float().sum().item() / result.size(0)


    @staticmethod
    def fn(result: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate the proportion of false negatives in the result tensor.

        Args:
            result (torch.Tensor): The predicted tensor.
            target (torch.Tensor): The ground truth tensor.

        Returns:
            float: The proportion of false negatives, calculated as the number of 
                   elements where the result is 0 and the target is 1, divided by 
                   the total number of elements in the result tensor.
        """
        return ((result == 0) & (target == 1)).float().sum().item() / result.size(0)


    @staticmethod
    def tn(result: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate the true negative rate for a binary classification problem.

        Args:
            result (torch.Tensor): The predicted labels as a tensor.
            target (torch.Tensor): The ground truth labels as a tensor.

        Returns:
            float: The true negative rate, which is the proportion of true negatives 
                   out of the total number of predictions.
        """
        return ((result == 0) & (target == 0)).float().sum().item() / result.size(0)


    @staticmethod
    def precision(result: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate the precision of the given results compared to the target.

        Precision is defined as the ratio of true positives (tp) to the sum of true positives and false positives (fp).

        Args:
            result (torch.Tensor): The predicted results.
            target (torch.Tensor): The ground truth labels.

        Returns:
            float: The precision score. Returns 0 if the denominator is 0.
        """
        tp = MeasureToolCNN.tp(result, target)
        fp = MeasureToolCNN.fp(result, target)
        denominator = tp + fp
        return tp / denominator if denominator != 0 else 0


    @staticmethod
    def sensitivity(result: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate the sensitivity (true positive rate) of the given results.

        Sensitivity is defined as the ratio of true positives (tp) to the sum of true positives and false negatives (fn).

        Args:
            result (torch.Tensor): The predicted results.
            target (torch.Tensor): The ground truth labels.

        Returns:
            float: The sensitivity value. Returns 0 if the denominator is 0.
        """
        tp = MeasureToolCNN.tp(result, target)
        fn = MeasureToolCNN.fn(result, target)
        denominator = tp + fn
        return tp / denominator if denominator != 0 else 0


    @staticmethod
    def specificity(result: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate the specificity metric for binary classification.

        Specificity, also known as the true negative rate, measures the proportion of actual negatives that are correctly identified as such.

        Args:
            result (torch.Tensor): The predicted labels as a tensor.
            target (torch.Tensor): The ground truth labels as a tensor.

        Returns:
            float: The specificity score. Returns 0 if the denominator is 0.
        """
        tn = MeasureToolCNN.tn(result, target)
        fp = MeasureToolCNN.fp(result, target)
        denominator = tn + fp
        return tn / denominator if denominator != 0 else 0


    @staticmethod
    def npv(result: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate the Negative Predictive Value (NPV) for the given result and target tensors.

        NPV is defined as the ratio of true negatives (TN) to the sum of true negatives (TN) and false negatives (FN).

        Args:
            result (torch.Tensor): The predicted values tensor.
            target (torch.Tensor): The ground truth values tensor.

        Returns:
            float: The Negative Predictive Value (NPV). Returns 0 if the denominator is 0.
        """
        tn = MeasureToolCNN.tn(result, target)
        fn = MeasureToolCNN.fn(result, target)
        denominator = tn + fn
        return tn / denominator if denominator != 0 else 0


    @staticmethod
    def fpr(result: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate the False Positive Rate (FPR) given the prediction results and the target labels.

        The False Positive Rate is defined as the ratio of false positives (FP) to the sum of false positives (FP) and true negatives (TN).

        Args:
            result (torch.Tensor): The predicted results as a tensor.
            target (torch.Tensor): The ground truth labels as a tensor.

        Returns:
            float: The calculated False Positive Rate. Returns 0 if the denominator is 0.
        """
        fp = MeasureToolCNN.fp(result, target)
        tn = MeasureToolCNN.tn(result, target)
        denominator = fp + tn
        return fp / denominator if denominator != 0 else 0


    @staticmethod
    def accuracy(result: torch.Tensor, target: torch.Tensor) -> float:
        """
        Computes the accuracy of the predictions.

        Args:
            result (torch.Tensor): The predicted labels.
            target (torch.Tensor): The true labels.

        Returns:
            float: The accuracy of the predictions as a float value between 0 and 1.
        """
        correct = (result == target).float().sum().item()
        total = target.numel()
        return correct / total if total != 0 else 0


    @staticmethod
    def fscore(result: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate the F1 score, which is the harmonic mean of precision and recall.

        Args:
            result (torch.Tensor): The predicted values.
            target (torch.Tensor): The ground truth values.

        Returns:
            float: The F1 score. Returns 0 if the denominator is 0 to avoid division by zero.
        """
        precision = MeasureToolCNN.precision(result, target)
        recall = MeasureToolCNN.sensitivity(result, target)
        numerator = 2 * (precision * recall)
        denominator = precision + recall
        return numerator / denominator if denominator != 0 else 0


    @staticmethod
    def iou(result: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate the Intersection over Union (IoU) between two tensors.

        Args:
            result (torch.Tensor): The predicted tensor, where each element is either 0 or 1.
            target (torch.Tensor): The ground truth tensor, where each element is either 0 or 1.

        Returns:
            float: The IoU score, which is the ratio of the intersection to the union of the two tensors.
                   Returns 0 if the union is 0 to avoid division by zero.
        """
        intersection = ((result == 1) & (target == 1)).float().sum().item()
        union = ((result == 1) | (target == 1)).float().sum().item()
        return intersection / union if union != 0 else 0


    @staticmethod
    def mcc(result: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute the Matthews correlation coefficient (MCC) for binary classification.

        The MCC is a measure of the quality of binary classifications, taking into account
        true and false positives and negatives. It returns a value between -1 and +1, where
        +1 indicates a perfect prediction, 0 indicates no better than random prediction, and
        -1 indicates total disagreement between prediction and observation.

        Args:
            result (torch.Tensor): The predicted labels as a tensor.
            target (torch.Tensor): The true labels as a tensor.

        Returns:
            float: The MCC value.
        """
        tp = MeasureToolCNN.tp(result, target)
        fp = MeasureToolCNN.fp(result, target)
        fn = MeasureToolCNN.fn(result, target)
        tn = MeasureToolCNN.tn(result, target)
        numerator = (tp * tn - fp * fn)
        denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return numerator / denominator if denominator != 0 else 0


    @staticmethod
    def auc(result: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute the Area Under the Receiver Operating Characteristic Curve (ROC AUC) 
        from prediction scores.

        Args:
            result (torch.Tensor): The predicted scores or probabilities.
            target (torch.Tensor): The ground truth binary labels.

        Returns:
            float: The computed ROC AUC score.
        """
        return roc_auc_score(target.flatten(), result.flatten())

    @staticmethod
    def ci(result: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate the confidence interval (CI) for a given result tensor.

        The CI is computed using the standard deviation of the result tensor,
        multiplied by 1.96 (for a 95% confidence level), and divided by the 
        square root of the number of elements in the result tensor.

        Args:
            result (torch.Tensor): The tensor containing the results.
            target (torch.Tensor): The tensor containing the target values (not used in the calculation).

        Returns:
            float: The calculated confidence interval.
        """
        std = result.float().std().item()
        ci = std * 1.96 / (result.float().numel() ** 0.5)
        return ci