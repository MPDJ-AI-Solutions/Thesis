import os

import cv2
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.dataset_info import TransformerModelSpectralImageInfo
from models.TransformerMethaneDetection.Backbone.backbone import Backbone
from models.TransformerMethaneDetection.Segmentation.segmentation import SegmentationModel
from models.TransformerMethaneDetection.Transformer.transformer import Transformer

from dataset.STARCOP_dataset import STARCOPDataset, DatasetType, SpectralImageInfo
from models.TransformerMethaneDetection.detector import Detector


if __name__ == "__main__":
    # Training hyperparameters
    batch_size = 8
    num_epochs = 10
    learning_rate = 1e-4

    # Initialize device (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the models
    backbone = Backbone(input_size=8, pretrained=True).to(device)
    transformer = Transformer(d_model=512, nhead=8, encoder_num_layers=6, num_decoder_layers=6).to(device)
    detector = Detector(backbone, transformer).to(device)
    mask_head = SegmentationModel(detector).to(device)

    # Initialize the Detector model

    # Load your dataset
    train_dataset = STARCOPDataset(data_path=r"data", data_type=DatasetType.TRAIN, image_info_class=TransformerModelSpectralImageInfo)
    val_dataset = STARCOPDataset(data_path=r"data", data_type=DatasetType.TEST, image_info_class=TransformerModelSpectralImageInfo)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Define the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # or another loss function suitable for your task
    optimizer = optim.Adam(detector.parameters(), lr=learning_rate)


    # Function to train the model
    def train_one_epoch(model, data_loader, optimizer, criterion, device):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            images, masks = TransformerModelSpectralImageInfo.backbone_input_converter(batch)
            bbox = TransformerModelSpectralImageInfo.get_bbox(batch)

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images, masks)

            # Compute loss
            #loss = criterion(outputs, masks)
            #loss.backward()

            # Update parameters
            #optimizer.step()

            #running_loss += loss.item()

        return running_loss / len(data_loader)


    # Function to evaluate the model on validation data
    def evaluate(model, data_loader, criterion, device):
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images, masks = TransformerModelSpectralImageInfo.backbone_input_converter(batch)

                # Transfer to cuda
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images, masks)

                loss = criterion(outputs, masks)
                val_loss += loss.item()

        return val_loss / len(data_loader)


    # Main training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Train for one epoch
        train_loss = train_one_epoch(detector, train_loader, optimizer, criterion, device)
        print(f"Training Loss: {train_loss}")

        # Evaluate on validation set
        val_loss = evaluate(detector, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss}")

        # Save the model checkpoint
        torch.save(detector.state_dict(), f'detector_epoch_{epoch + 1}.pth')

    print("Training complete.")