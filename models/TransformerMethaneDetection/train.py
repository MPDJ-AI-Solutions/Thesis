from datetime import datetime

import torch
from torch import optim
from torch.utils.data import DataLoader

from dataset.STARCOP_dataset import STARCOPDataset
from dataset.dataset_info import FilteredSpectralImageInfo
from dataset.dataset_type import DatasetType
from models.Tools.measures.measure_tool_factory import MeasureToolFactory
from models.Tools.measures.model_type import ModelType
from models.Tools.model_files_handler import ModelFilesHandler

from .model import TransformerModel


def train(
        model: TransformerModel,
        dataloader : DataLoader,
        criterion,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda"
    ):
    model.train()  # Set model to training mode
    running_loss = 0.0

    for batch_idx, (image, filtered_image, mag1c, labels_rgba, labels_binary, bboxes) in enumerate(dataloader):
        image = image.to(device)
        filtered_image = filtered_image.to(device)
        binary_mask = labels_binary.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # Currently only segmentation
        _, _, computed_mask = model(image, filtered_image)

        # Calculate loss (using Binary Cross Entropy) - use different to see how it impacts the model
        mask_loss = criterion(computed_mask.to(device), binary_mask.to(device))

        total_loss = mask_loss

        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

        # Log running loss (you can add more sophisticated logging if needed)
        running_loss += total_loss.item()

        # Print every 100 steps (you can adjust this)
        if batch_idx % 10 == 0:
            print(f"Batch [{batch_idx}/{len(dataloader)}], Loss: {running_loss / (batch_idx + 1):.4f}")

    return running_loss / len(dataloader)  # Return average loss for the epoch


def evaluate(model, dataloader, criterion, measurer, device="cuda"):
    model.eval()  # Set model to evaluation mode

    all_predictions: list = []
    all_targets: list = []
    running_loss: float = 0.0

    with torch.no_grad():
        for batch_idx, (image, filtered_image, mag1c, labels_rgba, labels_binary, bboxes) in enumerate(
                dataloader):
            image = image.to(device)
            filtered_image = filtered_image.to(device)
            target_mask = labels_binary.to(device)

            # Forward pass
            _, _, mask = model(image, filtered_image)

            # Calculate loss]
            binary_mask = (mask > 0.5).float()  # Example threshold, can be adjusted
            mask_loss = criterion(binary_mask, target_mask)

            # Total loss
            total_loss = mask_loss

            # Log running loss
            running_loss += total_loss.item()

            all_predictions.append(mask.cpu())
            all_targets.append(target_mask.cpu())

    measures = measurer.compute_measures(torch.cat(all_predictions), torch.cat(all_targets))

    return running_loss / len(dataloader), measures  # Return average loss for the epoch


# Initialize dataset and dataloader
if __name__ == "__main__":
    dataset_train = STARCOPDataset(
        data_path=r"data",
        data_type=DatasetType.TRAIN,
        image_info_class=FilteredSpectralImageInfo,
    )
    dataset_test = STARCOPDataset(
        data_path=r"data",
        data_type=DatasetType.TEST,
        image_info_class=FilteredSpectralImageInfo,
    )

    train_dataloader = DataLoader(dataset_train, batch_size=5, shuffle=True)
    test_dataloader = DataLoader(dataset_test, batch_size=5, shuffle=True)

    # Initialize the model
    d_model = 256
    transformer_model = TransformerModel(d_model=d_model).to("cuda")

    # Define the loss function and optimizer
    criterion = torch.nn.BCELoss()  # Example loss function (you can define custom losses)
    optimizer = optim.Adam(transformer_model.parameters(), lr=1e-5)

    measurer = MeasureToolFactory.get_measure_tool(ModelType.TRANSFORMER)

    # Training loop
    epochs = 10  # Set the number of epochs
    for epoch in range(epochs):
        print(f"Epoch [{epoch + 1}/{epochs}]")
        train_loss = train(
            model=transformer_model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device="cuda")
        print(f"Training Loss: {train_loss:.4f}")

        # Evaluate on the validation set (if you have one)
        val_loss, measures = evaluate(transformer_model, test_dataloader, criterion, measurer=measurer, device="cuda")

        print(f"Validation Loss: {val_loss:.4f}")
        print(measures)

        model_handler = ModelFilesHandler()
        model_handler.save_model(
            model=transformer_model,
            epoch=epoch,
            metrics=measures,
            model_type=ModelType.TRANSFORMER
        )