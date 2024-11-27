import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, AdamW, DetrConfig

from dataset.STARCOP_dataset import STARCOPDataset
from dataset.dataset_info import FilteredSpectralImageInfo
from dataset.dataset_type import DatasetType
from models.Tools.Measures.measure_tool_factory import MeasureToolFactory
from models.Tools.Measures.model_type import ModelType
from models.Tools.model_files_handler import ModelFilesHandler


def create_datasets():
    dataset_train = STARCOPDataset(
        data_path=r"data",
        data_type=DatasetType.TRAIN,
        image_info_class=FilteredSpectralImageInfo,
        normalization=False
    )

    dataset_test = STARCOPDataset(
        data_path=r"data",
        data_type=DatasetType.TEST,
        image_info_class=FilteredSpectralImageInfo,
        normalization=False
    )

    return dataset_train, dataset_test

class DetrWith9Channels(nn.Module):
    def __init__(self, detr_model_name="facebook/detr-resnet-50", num_channels=9):
        super().__init__()

        # Load pre-trained DETR model
        config = DetrConfig.from_pretrained(detr_model_name)
        config.num_labels = 2  # One foreground class + background
        config.num_queries = 10
        self.detr = DetrForObjectDetection(config=config)

        # Modify the first convolutional layer of the ResNet backbone to accept 9 channels
        resnet_backbone = self.detr.model.backbone
        conv_layer = resnet_backbone.conv_encoder.model.conv1

        # Original ResNet's first conv layer is:
        # nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Change input channels from 3 to 9
        conv_layer = nn.Conv2d(
            in_channels=num_channels,  # Change input channels to 9
            out_channels=conv_layer.out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            bias=conv_layer.bias
        )

        resnet_backbone.conv_encoder.model.conv1 = conv_layer
        self.detr.model.backbone= resnet_backbone

        for name, param in self.detr.model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        self.detr.model.backbone.conv_encoder.model.conv1.requires_grad = True

    def forward(self, pixel_values, labels):
        return self.detr(pixel_values, labels = labels)


if __name__ == "__main__":
    num_epochs = 1
    learning_rate = 1e-4
    batch_size = 4
    device = "cuda"

    model = DetrWith9Channels()
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    measurer = MeasureToolFactory.get_measure_tool(ModelType.TRANSFORMER)

    dataset_train, dataset_test = create_datasets()

    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0

        for batch_idx, (image, _, mag1c, _, _, bboxes, labels) in enumerate(train_dataloader):
            image = image.to(device)
            mag1c = mag1c.to(device)
            input_data = torch.cat((image, mag1c), dim=1)

            target_bboxes = bboxes.to(device)
            target_labels = labels.to(device)

            targets = [
                {"boxes": target_bboxes[i], "class_labels": target_labels[i]}
                for i in range(target_bboxes.size(0))
            ]

            optimizer.zero_grad()

            outputs = model(pixel_values=input_data, labels=targets)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            print(f"Batch number: {batch_idx}, Loss: {loss}")



        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader)}")

    measures = measurer.compute_measures(torch.rand(100, 100), torch.rand(100, 100))

    model_handler = ModelFilesHandler()
    model_handler.save_model(
        model=model,
        epoch=10,
        metrics=measures,
        model_type=ModelType.TRANSFORMER
    )