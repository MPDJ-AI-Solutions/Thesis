import sys
import torch

from typing import Type
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from dataset.dataset_info import ClassifierDatasetInfo, DatasetInfo
from dataset.dataset_type import DatasetType
from dataset.STARCOP_dataset import STARCOPDataset


def print_progress_bar(percentage, loss):
    bar_length = 50  # Length of the progress bar
    filled_length = int(bar_length * percentage // 100)
    bar = '=' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write(f"\r[{bar}] {percentage:.2f}% [Loss: {loss:.6f}]")
    sys.stdout.flush()


def setup_dataloaders(
        data_path: str = r"data",
        batch_size: int = 32,
        train_type = DatasetType.EASY_TRAIN,
        image_info_class: Type[DatasetInfo] = ClassifierDatasetInfo,
        crop_size: int = 1
):
    train_dataset = STARCOPDataset(
        data_path=data_path,
        data_type=train_type,
        image_info_class=image_info_class,
        crop_size=crop_size
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = STARCOPDataset(
        data_path=data_path,
        data_type=DatasetType.TEST,
        image_info_class=image_info_class,
        crop_size=crop_size
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader


def setup_model(model: nn.Module, lr: float, device: str):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()  # Binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, criterion, optimizer


def train(criterion, device, epochs, model, optimizer, dataloader, model_handler, log_batches: bool = False):
    model.train()
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        running_loss = 0.0
        for batch_id, (images, mag1c, labels) in enumerate(dataloader):
            optimizer.zero_grad()

            input_image = torch.cat((images, mag1c), dim=1).to(device)
            labels = labels.long().to(device)

            outputs = model(input_image)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if log_batches and (batch_id + 1) % 10 == 0:
                print_progress_bar(batch_id / len(dataloader) * 100, running_loss / (batch_id + 1))

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")
        model_handler.save_raw_model(model)


def evaluate(criterion, device, model, dataloader, measurer):
    model.eval()
    all_predictions = []
    all_labels = []
    running_loss = 0.0

    for batch_id, (images, mag1c, labels) in enumerate(dataloader):
        input_image = torch.cat((images, mag1c), dim=1).to(device)
        labels = labels.long().to(device)

        outputs = model(input_image)
        predictions = torch.argmax(outputs, dim=1)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        all_predictions.append(predictions.cpu().detach())
        all_labels.append(labels.cpu().detach())

    measures = measurer.compute_measures(torch.cat(all_predictions), torch.cat(all_labels))
    print(f"Validation loss: {running_loss / len(dataloader)}.\nMeasures:\n{measures}")
    return measures


def setup_dataloaders_with_cross_validation(
        data_path: str = r"data",
        batch_size: int = 32,
        n_splits: int = 5,
        train_type = DatasetType.EASY_TRAIN,
        image_info_class: Type[DatasetInfo] = ClassifierDatasetInfo,
        crop_size: int = 1
):
    full_dataset = STARCOPDataset(
        data_path=data_path,
        data_type=train_type,
        image_info_class=image_info_class,
        crop_size=crop_size
    )

    # KFold splits
    kfold = KFold(n_splits=n_splits, shuffle=True)
    train_dataloaders = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        train_dataloaders.append((train_dataloader, val_dataloader))

    test_dataset = STARCOPDataset(
        data_path=data_path,
        data_type=DatasetType.TEST,
        image_info_class=image_info_class,
        crop_size=crop_size
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloaders, test_dataloader