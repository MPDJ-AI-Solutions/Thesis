import torch
from scipy.optimize import linear_sum_assignment


class HungarianMatcher:
    def __init__(self, bbox_cost: float = 1.0, confidence_cost: float = 1.0, iou_cost: float = 1.0):
        self.bbox_cost = bbox_cost
        self.confidence_cost = confidence_cost
        self.iou_cost = iou_cost

    @staticmethod
    def compute_iou(pred_boxes, target_boxes):
        """Compute IoU between predicted and target boxes."""
        x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        area_target = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        union = area_pred + area_target - intersection
        iou = intersection / (union + 1e-6)
        return iou

    def match(self, pred_boxes, target_boxes, pred_confidence=None, target_confidence=None):
        """Perform Hungarian matching."""
        batch_size = pred_boxes.size(0)

        # Initialize empty lists to store indices
        all_pred_indices = []
        all_target_indices = []

        for i in range(batch_size):
            # For each batch item, compute the costs
            pred_box = pred_boxes[i]
            target_box = target_boxes[i]

            # Compute IoU cost
            iou = self.compute_iou(pred_box, target_box)
            iou_cost = -iou  # Higher IoU should have lower cost

            # Compute bbox regression cost (L1 distance)
            bbox_cost = torch.cdist(pred_box, target_box, p=1)

            # Combine costs
            cost_matrix = self.bbox_cost * bbox_cost + self.iou_cost * iou_cost

            # Add confidence cost if available
            if pred_confidence is not None and target_confidence is not None:
                pred_conf = pred_confidence[i]
                target_conf = target_confidence[i]
                confidence_cost = torch.cdist(pred_conf, target_conf, p=1)
                cost_matrix += self.confidence_cost * confidence_cost

            # Solve assignment problem (Hungarian algorithm)
            pred_indices, target_indices = linear_sum_assignment(cost_matrix.cpu().detach().numpy())

            # Store indices for this batch
            all_pred_indices.append(torch.tensor(pred_indices))
            all_target_indices.append(torch.tensor(target_indices))

        # Return indices as tensors
        return torch.stack(all_pred_indices), torch.stack(all_target_indices)
