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
    """
    Prints a progress bar to the console.

    The progress bar is displayed in the format:
    [====================----------] 40.00% [Loss: 0.123456]

    Args:
        percentage (float): The current progress as a percentage (0 to 100).
        loss (float): The current loss value to display.   
    """
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
    """
    Sets up the data loaders for training and testing datasets.
    
    Args:
        data_path (str): Path to the data directory. Default is "data".
        batch_size (int): Number of samples per batch to load. Default is 32.
        train_type (DatasetType): Type of the training dataset. Default is DatasetType.EASY_TRAIN.
        image_info_class (Type[DatasetInfo]): Class for loading image. Default is ClassifierDatasetInfo.
        crop_size (int): Size of the crop to be applied to the images. Default is 1.
    Returns:
        tuple: A tuple containing the training DataLoader and the testing DataLoader.
    """
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
    """
    Set up the model for training by moving it to the specified device, 
    and initializing the loss criterion and optimizer.
    
    Args:
        model (nn.Module): The neural network model to be trained.
        lr (float): The learning rate for the optimizer.
        device (str): The device to which the model should be moved ('cpu' or 'cuda').
    Returns:
        tuple: A tuple containing the model moved to the specified device, 
               the loss criterion (CrossEntropyLoss), and the optimizer (Adam).
    """
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()  # Binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, criterion, optimizer


def train(criterion, device, epochs, model, optimizer, dataloader, model_handler, log_batches: bool = False):
    """
    Trains a given model using the specified criterion, optimizer, and dataloader. It saves raw model after training. 
    
    Args:
        criterion (torch.nn.Module): The loss function to be used.
        device (torch.device): The device to run the training on (e.g., 'cpu' or 'cuda').
        epochs (int): The number of epochs to train the model for.
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer to use for updating the model parameters.
        dataloader (torch.utils.data.DataLoader): The dataloader providing the training data.
        model_handler (object): An object responsible for handling model saving.
        log_batches (bool, optional): If True, logs progress every 10 batches. Default is False.
    """
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
    """
    Evaluates the performance of a model on a given dataset.

    Args:
        criterion (torch.nn.Module): The loss function used to compute the loss.
        device (torch.device): The device (CPU or GPU) on which the computation will be performed.
        model (torch.nn.Module): The model to be evaluated.
        dataloader (torch.utils.data.DataLoader): The DataLoader providing the evaluation data.
        measurer (object): An object with a method `compute_measures` that calculates performance metrics.
    Returns:
        dict: A dictionary containing the computed performance measures.
    """
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
    """
    Sets up dataloaders for training and validation using K-Fold cross-validation, 
    and a dataloader for testing.

    Args:
        data_path (str): Path to the dataset directory. Default is "data".
        batch_size (int): Number of samples per batch to load. Default is 32.
        n_splits (int): Number of folds for K-Fold cross-validation. Default is 5.
        train_type (DatasetType): Type of dataset to use for training. Default is DatasetType.EASY_TRAIN.
        image_info_class (Type[DatasetInfo]): Class to use for image loading. Default is ClassifierDatasetInfo.
        crop_size (int): Size of the crop to apply to images. Default is 1.
    Returns:
        tuple: A tuple containing:
            - train_dataloaders (list): A list of tuples, each containing a training dataloader and a validation dataloader for each fold.
            - test_dataloader (DataLoader): A dataloader for the test dataset.
        """
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