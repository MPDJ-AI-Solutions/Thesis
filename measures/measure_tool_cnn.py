from .measure_tool import MeasureTool

import torch
import math

class MeasureToolCNN(MeasureTool):
    @staticmethod
    def tp(result: torch.Tensor, target: torch.Tensor) -> float:
        return ((result == 1) & (target == 1)).float().sum().item() / result.size(0)


    @staticmethod
    def fp(result: torch.Tensor, target: torch.Tensor) -> float:
        return ((result == 1) & (target == 0)).float().sum().item() / result.size(0)


    @staticmethod
    def fn(result: torch.Tensor, target: torch.Tensor) -> float:
        return ((result == 0) & (target == 1)).float().sum().item() / result.size(0)


    @staticmethod
    def tn(result: torch.Tensor, target: torch.Tensor) -> float:
        return ((result == 0) & (target == 0)).float().sum().item() / result.size(0)


    @staticmethod
    def precision(result: torch.Tensor, target: torch.Tensor) -> float:
        tp = MeasureToolCNN.tp(result, target)
        fp = MeasureToolCNN.fp(result, target)
        denominator = tp + fp
        return tp / denominator if denominator != 0 else 0


    @staticmethod
    def sensitivity(result: torch.Tensor, target: torch.Tensor) -> float:
        tp = MeasureToolCNN.tp(result, target)
        fn = MeasureToolCNN.fn(result, target)
        denominator = tp + fn
        return tp / denominator if denominator != 0 else 0


    @staticmethod
    def specificity(result: torch.Tensor, target: torch.Tensor) -> float:
        tn = MeasureToolCNN.tn(result, target)
        fp = MeasureToolCNN.fp(result, target)
        denominator = tn + fp
        return tn / denominator if denominator != 0 else 0


    @staticmethod
    def npv(result: torch.Tensor, target: torch.Tensor) -> float:
        tn = MeasureToolCNN.tn(result, target)
        fn = MeasureToolCNN.fn(result, target)
        denominator = tn + fn
        return tn / denominator if denominator != 0 else 0


    @staticmethod
    def fpr(result: torch.Tensor, target: torch.Tensor) -> float:
        fp = MeasureToolCNN.fp(result, target)
        tn = MeasureToolCNN.tn(result, target)
        denominator = fp + tn
        return fp / denominator if denominator != 0 else 0


    @staticmethod
    def accuracy(result: torch.Tensor, target: torch.Tensor) -> float:
        correct = (result == target).float().sum()
        total = target.numel()
        return correct / total if total != 0 else 0


    @staticmethod
    def fscore(result: torch.Tensor, target: torch.Tensor) -> float:
        precision = MeasureToolCNN.precision(result, target)
        recall = MeasureToolCNN.sensitivity(result, target)
        numerator = 2 * (precision * recall)
        denominator = precision + recall
        return numerator / denominator if denominator != 0 else 0


    @staticmethod
    def iou(result: torch.Tensor, target: torch.Tensor) -> float:
        intersection = ((result == 1) & (target == 1)).float().sum()
        union = ((result == 1) | (target == 1)).float().sum()
        return intersection / union if union != 0 else 0


    @staticmethod
    def mcc(result: torch.Tensor, target: torch.Tensor) -> float:
        tp = MeasureToolCNN.tp(result, target)
        fp = MeasureToolCNN.fp(result, target)
        fn = MeasureToolCNN.fn(result, target)
        tn = MeasureToolCNN.tn(result, target)
        numerator = (tp * tn - fp * fn)
        denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return numerator / denominator if denominator != 0 else 0


    @staticmethod
    def auc(result: torch.Tensor, target: torch.Tensor) -> float:
        pass #placeholder

    # @staticmethod
    # def auc(result: torch.Tensor, target: torch.Tensor) -> float:
    #     predictions = result.cpu().detach()
    #     targets = target.cpu().detach()
    #
    #     # Compute False Positive Rate (fpr) and True Positive Rate (tpr)
    #     fpr, tpr, _ = roc(predictions, targets, pos_label=1)
    #
    #     # Compute AUC
    #     auc_score = auc(fpr, tpr)
    #     return auc_score.item()


    @staticmethod
    def ci(result: torch.Tensor, target: torch.Tensor) -> float:
        pass  # placeholder
    # @staticmethod
    # def ci(result: torch.Tensor, target: torch.Tensor) -> float:
    #     predictions = result.cpu().detach().numpy()
    #     targets = target.cpu().detach().numpy()
    #     num_bootstrap: int = 1000
    #     confidence: float = 0.95
    #     auc_scores = []
    #     n = len(predictions)
    #
    #     for _ in range(num_bootstrap):
    #         # Sample with replacement
    #         indices = np.random.choice(range(n), n, replace=True)
    #         sample_predictions = predictions[indices]
    #         sample_targets = targets[indices]
    #
    #         # Calculate AUC for the bootstrap sample
    #         auc_score = calculate_auc(torch.tensor(sample_predictions), torch.tensor(sample_targets))
    #         auc_scores.append(auc_score)
    #
    #     # Calculate confidence interval
    #     lower_bound = np.percentile(auc_scores, (1 - confidence) / 2 * 100)
    #     upper_bound = np.percentile(auc_scores, (1 + confidence) / 2 * 100)
    #
    #     return lower_bound, upper_bound