import torch
from torch import optim
from torch.nn import BCEWithLogitsLoss, SmoothL1Loss
from torch.utils.data import DataLoader

from dataset.STARCOP_dataset import STARCOPDataset
from dataset.dataset_info import SegmentationDatasetInfo
from dataset.dataset_type import DatasetType
from measures import MeasureToolFactory
from measures import ModelType
from files_handler import ModelFilesHandler
from models.Transformer.MethaneMapper.Segmentation.dice_loss import DiceLoss
from models.Transformer.MethaneMapper.Segmentation.hungarian_matcher import HungarianMatcher

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
    """
    Train the Transformer model for one epoch.
  
    Args:
        model (TransformerModel): The Transformer model to be trained.
        dataloader (DataLoader): DataLoader providing the training data.
        criterion_bbox: Loss function for bounding box regression.
        criterion_confidence: Loss function for confidence prediction.
        criterion_mask: Loss function for mask prediction (currently not used).
        matcher: Matcher object to match predicted and target bounding boxes and confidences.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        epoch (int): Current epoch number.
        max_epoch (int): Total number of epochs.
        device (str, optional): Device to run the training on. Default is "cuda".
    Returns:
        float: Average loss for the epoch.
    """
    model.train()  # Set model to training mode
    running_loss = 0.0
    running_loss_bbox = 0.0
    running_loss_confidence = 0.0

    for batch_idx, (image, filtered_image, mag1c, labels_rgba, labels_binary, bboxes, confidence) in enumerate(
            dataloader):
        image = image.to(device)
        filtered_image = filtered_image.to(device)
        binary_mask = labels_binary.to(device)
        target_bboxes = bboxes.to(device)
        target_confidence = confidence.to(device)
        mag1c = mag1c.to(device)

        # Identify valid samples with leakage (at least one '1' in the binary mask)
        valid_indices = (binary_mask.view(binary_mask.size(0), -1).sum(dim=1) > 0).nonzero(as_tuple=True)[0]

        if len(valid_indices) == 0:
            # Skip this batch if no samples have leakage
            continue

        # Filter the batch to include only valid samples
        image = image[valid_indices]
        filtered_image = filtered_image[valid_indices]
        binary_mask = binary_mask[valid_indices]
        target_bboxes = target_bboxes[valid_indices]
        target_confidence = target_confidence[valid_indices]
        mag1c = mag1c[valid_indices]

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # Currently only segmentation
        pred_bbox, pred_confidence = model(image, filtered_image, mag1c)

        pred_indices, target_indices = matcher.match(pred_bbox, target_bboxes, pred_confidence, target_confidence)

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
        total_loss.backward()

        optimizer.step()

        # Log running loss (you can add more sophisticated logging if needed)
        running_loss += total_loss.sum()

        running_loss_bbox += bbox_loss.sum()
        running_loss_confidence += confidence_loss.sum()


        if batch_idx % 10 == 0:
            print(
                f"Progress: {(epoch * len(dataloader) + batch_idx) / (len(dataloader) * max_epoch) * 100:.2f}%",
                f"Batch [{batch_idx}/{len(dataloader)}] ({(batch_idx / len(dataloader) * 100):2.2f}%),\n"
                f"Loss: total: {running_loss / (batch_idx + 1):.4f}, bbox: {running_loss_bbox / (batch_idx + 1):.4f}, confidence: {running_loss_confidence  / (batch_idx + 1):.4f}"
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
    """
    Evaluate the performance of the model on the given dataloader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the evaluation data.
        criterion_bbox (callable): Loss function for bounding box regression.
        criterion_confidence (callable): Loss function for confidence prediction.
        criterion_mask (callable): Loss function for mask prediction.
        matcher (object): Object responsible for matching predicted and target bounding boxes.
        measurer (object): Object responsible for computing evaluation metrics.
        device (str, optional): Device to run the evaluation on. Default is "cuda".
    Returns:
        float: Average loss over the evaluation dataset.
        dict: Dictionary containing evaluation metrics.
    """
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
            binary_mask = labels_binary.to(device)
            target_bboxes = bboxes.to(device)
            target_confidence = confidences.to(device)
            mag1c = mag1c.to(device)

            # Identify valid samples with leakage (at least one '1' in the binary mask)
            valid_indices = (binary_mask.view(binary_mask.size(0), -1).sum(dim=1) > 0).nonzero(as_tuple=True)[0]

            if len(valid_indices) == 0:
                # Skip this batch if no samples have leakage
                continue

            # Filter the batch to include only valid samples
            image = image[valid_indices]
            filtered_image = filtered_image[valid_indices]
            binary_mask = binary_mask[valid_indices]
            bboxes = target_bboxes[valid_indices]
            confidences = target_confidence[valid_indices]
            mag1c = mag1c[valid_indices]


            # Forward pass
            pred_bboxes, pred_confidences = model(image, filtered_image, mag1c)
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

            all_targets.append(binary_mask.cpu())


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
        data_type=DatasetType.EASY_TRAIN,
        image_info_class=SegmentationDatasetInfo,
        normalization=False
    )
    dataset_test = STARCOPDataset(
        data_path=r"data",
        data_type=DatasetType.TEST,
        image_info_class=SegmentationDatasetInfo,
        normalization=False
    )

    train_dataloader = DataLoader(dataset_train, batch_size=48, shuffle=True)
    test_dataloader = DataLoader(dataset_test, batch_size=48, shuffle=True)

    # Initialize the model
    d_model = 512
    transformer_model = TransformerModel(d_model=d_model, n_queries=1).to("cuda")
    for name, param in transformer_model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")
    # Define the loss function and optimizer
    criterion_mask = DiceLoss()
    criterion_bbox = SmoothL1Loss()
    criterion_confidence = BCEWithLogitsLoss()

    matcher = HungarianMatcher(bbox_cost=2.0, confidence_cost=1.0, iou_cost=2.0)

    optimizer = optim.Adam(transformer_model.parameters(), lr=1e-4)

    measurer = MeasureToolFactory.get_measure_tool(ModelType.TRANSFORMER)

    # Training loop
    epochs = 25  # Set the number of epochs
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
