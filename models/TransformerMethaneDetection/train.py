from datetime import datetime

import torch
from torch import optim, nn
from torch.nn import BCEWithLogitsLoss, BCELoss, SmoothL1Loss, L1Loss
from torch.utils.data import DataLoader

from dataset.STARCOP_dataset import STARCOPDataset
from dataset.dataset_info import FilteredSpectralImageInfo
from dataset.dataset_type import DatasetType
from models.Tools.Measures.measure_tool_factory import MeasureToolFactory
from models.Tools.Measures.model_type import ModelType
from models.Tools.model_files_handler import ModelFilesHandler
from models.TransformerMethaneDetection.dice_loss import DiceLoss
from models.TransformerMethaneDetection.hungarian_matcher import HungarianMatcher

from .model import TransformerModel


def train(
        model: TransformerModel,
        dataloader: DataLoader,
        criterion_bbox,
        criterion_confidence,
        criterion_mask,
        matcher,
        optimizer: torch.optim.Optimizer,
        epoch:int,
        max_epoch:int,
        device: str = "cuda",
):
    model.train()  # Set model to training mode
    running_loss = 0.0
    running_loss_bbox = 0.0
    running_loss_confidence = 0.0
    running_loss_mask = 0.0

    for batch_idx, (image, filtered_image, mag1c, labels_rgba, labels_binary, bboxes, confidence) in enumerate(
            dataloader):
        image = image.to(device)
        filtered_image = filtered_image.to(device)
        binary_mask = labels_binary.to(device)
        target_bboxes = bboxes.to(device)
        target_confidence = confidence.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # Currently only segmentation
        pred_bbox, pred_confidence = model(image, filtered_image)

        pred_indices, target_indices = matcher.match(pred_bbox, target_bboxes, pred_confidence, pred_confidence)

        # Extract matched predictions and targets
        matched_pred_boxes = []
        matched_target_boxes = []
        matched_pred_confidences = []
        matched_target_confidences = []

        # Loop over each sample in the batch
        for i in range(pred_indices.size(0)):  # Loop over batch size (5)
            # Extract the indices for the current sample in the batch
            pred_idx = pred_indices[i]  # Shape (100,)
            target_idx = target_indices[i]  # Shape (100,)

            # Use the indices to gather the matching boxes and confidences
            matched_pred_boxes.append(pred_bbox[i][pred_idx])  # Shape (100, 4)
            matched_target_boxes.append(target_bboxes[i][target_idx])  # Shape (100, 4)
            matched_pred_confidences.append(pred_confidence[i][pred_idx])  # Shape (100,)
            matched_target_confidences.append(target_confidence[i][target_idx])  # Shape (100,)

        # Convert lists to tensors
        matched_pred_boxes = torch.stack(matched_pred_boxes)
        matched_target_boxes = torch.stack(matched_target_boxes)
        matched_pred_confidences = torch.stack(matched_pred_confidences)
        matched_target_confidences = torch.stack(matched_target_confidences)

        # Criterion for bbox regression
        bbox_loss = criterion_bbox(matched_pred_boxes, matched_target_boxes)
        # Criterion for confidence prediction
        confidence_loss = criterion_confidence(matched_pred_confidences, matched_target_confidences)


        total_loss = bbox_loss + confidence_loss
        total_loss.sum().backward()
        optimizer.step()

        # Log running loss (you can add more sophisticated logging if needed)
        running_loss += total_loss.sum()

        running_loss_bbox += bbox_loss.sum()
        running_loss_confidence += confidence_loss.sum()


        if batch_idx % 10 == 0:
            print(
                f"Progress: {epoch * batch_idx + batch_idx}/{len(dataloader) * max_epoch}%",
                f"Batch [{batch_idx}/{len(dataloader)}] ({(batch_idx / len(dataloader) * 100):2.2f}%),\n"
                f"Loss: \n total: {running_loss:.4f}\n bbox: {running_loss_bbox:.4f}\n confidence: {running_loss_confidence:.4f}"
            )

    return running_loss / len(dataloader)  # Return average loss for the epoch


def evaluate(
        model,
        dataloader,
        criterion_bbox,
        criterion_confidence,
        criterion_mask,
        matcher,
        measurer,
        device="cuda"
):
    model.eval()  # Set model to evaluation mode

    all_predictions: list = []
    all_targets: list = []
    running_loss: float = 0.0
    running_loss_bbox = 0.0
    running_loss_confidence = 0.0
    running_loss_mask = 0.0

    with torch.no_grad():
        for batch_idx, (image, filtered_image, mag1c, labels_rgba, labels_binary, bboxes, confidences) in enumerate(
                dataloader):
            image = image.to(device)
            filtered_image = filtered_image.to(device)
            target_mask = labels_binary.to(device)
            bboxes = bboxes.to(device)
            confidences = confidences.to(device)

            # Forward pass
            pred_bboxes, pred_confidences, mask = model(image, filtered_image)
            pred_indices, target_indices = matcher.match(pred_bboxes, bboxes, pred_confidences,
                                                         confidences)

            # Extract matched predictions and targets
            matched_pred_boxes = []
            matched_target_boxes = []
            matched_pred_confidences = []
            matched_target_confidences = []

            # Loop over each sample in the batch
            for i in range(pred_indices.size(0)):  # Loop over batch size (5)
                # Extract the indices for the current sample in the batch
                pred_idx = pred_indices[i]  # Shape (100,)
                target_idx = target_indices[i]  # Shape (100,)

                # Use the indices to gather the matching boxes and confidences
                matched_pred_boxes.append(pred_bboxes[i][pred_idx])  # Shape (100, 4)
                matched_target_boxes.append(bboxes[i][target_idx])  # Shape (100, 4)
                matched_pred_confidences.append(pred_confidences[i][pred_idx])  # Shape (100,)
                matched_target_confidences.append(confidences[i][target_idx])  # Shape (100,)

            # Convert lists to tensors
            matched_pred_boxes = torch.stack(matched_pred_boxes)
            matched_target_boxes = torch.stack(matched_target_boxes)
            matched_pred_confidences = torch.stack(matched_pred_confidences)
            matched_target_confidences = torch.stack(matched_target_confidences)

            # Criterion for bbox regression
            bbox_loss = criterion_bbox(matched_pred_boxes, matched_target_boxes)
            # Criterion for confidence prediction
            confidence_loss = criterion_confidence(matched_pred_confidences, matched_target_confidences)

            # Total loss
            total_loss = bbox_loss + confidence_loss

            # Log running loss
            running_loss += total_loss.item()
            running_loss_bbox += bbox_loss.sum()
            running_loss_confidence += confidence_loss.sum()

            #all_predictions.append(binary_mask.cpu())
            all_targets.append(target_mask.cpu())


    measures = measurer.compute_measures(torch.cat(all_targets), torch.cat(all_targets))

    print(
        f"Validation loss: {running_loss / (batch_idx + 1):.4f}\n",
        f"Confidence loss: {running_loss_confidence / (batch_idx + 1):.4f}\n",
        f"BBOX loss: {running_loss_bbox / (batch_idx + 1):.4f}\n",
    )
    return running_loss / len(dataloader), measures  # Return average loss for the epoch


# Initialize dataset and dataloader
if __name__ == "__main__":
    dataset_train = STARCOPDataset(
        data_path=r"data",
        data_type=DatasetType.TRAIN,
        image_info_class=FilteredSpectralImageInfo,
        enable_augmentation=False
    )
    dataset_test = STARCOPDataset(
        data_path=r"data",
        data_type=DatasetType.TEST,
        image_info_class=FilteredSpectralImageInfo,
        enable_augmentation=False
    )

    train_dataloader = DataLoader(dataset_train, batch_size=5, shuffle=True)
    test_dataloader = DataLoader(dataset_test, batch_size=5, shuffle=True)

    # Initialize the model
    d_model = 1024
    transformer_model = TransformerModel(d_model=d_model, n_queries=16).to("cuda")

    # Define the loss function and optimizer
    criterion_mask = DiceLoss()
    criterion_bbox = L1Loss()
    criterion_confidence = L1Loss()

    matcher = HungarianMatcher(bbox_cost=1.0, confidence_cost=1.0, iou_cost=2.0)

    optimizer = optim.Adam(transformer_model.parameters(), lr=1e-5)

    measurer = MeasureToolFactory.get_measure_tool(ModelType.TRANSFORMER)

    # Training loop
    epochs = 10  # Set the number of epochs
    for epoch in range(epochs):
        print(f"Epoch [{epoch + 1}/{epochs}]")
        train_loss = train(
            model=transformer_model,
            dataloader=train_dataloader,
            criterion_mask=criterion_mask,
            criterion_bbox=criterion_bbox,
            criterion_confidence=criterion_confidence,
            matcher=matcher,
            optimizer=optimizer,
            device="cuda",
            epoch=epoch,
            max_epoch=epochs
        )
        print(f"Training Loss: {train_loss:.4f}")

        # Evaluate on the validation set (if you have one)
        val_loss, measures = evaluate(
            transformer_model,
            test_dataloader,
            criterion_mask=criterion_mask,
            criterion_bbox=criterion_bbox,
            criterion_confidence=criterion_confidence,
            matcher=matcher,
            measurer=measurer,
            device="cuda"
        )

        print(f"Validation Loss: {val_loss:.4f}")
        print(measures)

        model_handler = ModelFilesHandler()
        model_handler.save_model(
            model=transformer_model,
            epoch=epoch,
            metrics=measures,
            model_type=ModelType.TRANSFORMER
        )
