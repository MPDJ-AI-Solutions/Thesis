from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.dataset_info import ClassifierDatasetInfo
from dataset.dataset_type import DatasetType
from dataset.STARCOP_dataset import STARCOPDataset



def setup_dataloaders(data_path: str = r"data", batch_size: int = 32, train_type = DatasetType.EASY_TRAIN):
    train_dataset = STARCOPDataset(
        data_path=data_path,
        data_type=train_type,
        image_info_class=ClassifierDatasetInfo,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = STARCOPDataset(
        data_path=data_path,
        data_type=DatasetType.TEST,
        image_info_class=ClassifierDatasetInfo,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader


def setup_model(model: nn.Module, lr: float, device: str):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()  # Binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, criterion, optimizer


def train(criterion, device, epochs, model, optimizer, dataloader, transform: Optional[transforms] = None, log_batches: bool = False):
    model.train()
    for epoch in range(epochs):  # Adjust the number of epochs
        running_loss = 0.0
        for batch_id, (images, mag1c, labels) in enumerate(dataloader):  # Assume a PyTorch DataLoader is set up
            optimizer.zero_grad()

            input_image = torch.cat((images, mag1c), dim=1)
            labels = labels.long().to(device)

            outputs = model((transform(input_image) if transform else  input_image).to(device))

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if log_batches and (batch_id + 1) % 10 == 0:
                print(f"Batch: {batch_id + 1}, Loss: {running_loss / (batch_id + 1)}")

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")


def evaluate(criterion, device, model, dataloader, measurer, transform: Optional[transforms] = None):
    model.eval()
    all_predictions = []
    all_labels = []
    running_loss = 0.0

    for batch_id, (images, mag1c, labels) in enumerate(dataloader):
        input_image = torch.cat((images, mag1c), dim=1)
        labels = labels.long().to(device)

        outputs = model((transform(input_image) if transform else  input_image).to(device))
        predictions = torch.argmax(outputs, dim=1)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        all_predictions.append(predictions.cpu().detach())
        all_labels.append(labels.cpu().detach())

    measures = measurer.compute_measures(torch.cat(all_predictions), torch.cat(all_labels))
    print(f"Validation loss: {running_loss / len(dataloader)}.\nMeasures:\n{measures}")
    return measures