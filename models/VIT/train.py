import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from dataset.dataset_info import ClassifierSpectralImageInfo
from dataset.dataset_type import DatasetType
from dataset.STARCOP_classifier_dataset import STARCOPDataset

from models.Tools.measures.measure_tool_factory import MeasureToolFactory
from models.Tools.measures.model_type import ModelType
from models.Tools.model_files_handler import ModelFilesHandler
from models.VIT.model import CustomViT


def setup_dataloaders(data_path: str = r"data", batch_size: int = 32):
    train_dataset = STARCOPDataset(
        data_path=data_path,
        data_type=DatasetType.EASY_TRAIN,
        image_info_class=ClassifierSpectralImageInfo,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = STARCOPDataset(
        data_path=data_path,
        data_type=DatasetType.TEST,
        image_info_class=ClassifierSpectralImageInfo,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader


def setup_model(lr: float, device: str):
    model = CustomViT().to(device)

    criterion = nn.CrossEntropyLoss()  # Binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, criterion, optimizer


def train(criterion, device, epochs, model, optimizer, dataloader, transform):
    model.train()
    for epoch in range(epochs):  # Adjust the number of epochs
        running_loss = 0.0
        for batch_id, (images, mag1c, labels) in enumerate(dataloader):  # Assume a PyTorch DataLoader is set up
            optimizer.zero_grad()

            input_image = torch.cat((images, mag1c), dim=1)
            input_image = transform(input_image).to(device)
            labels = labels.long().to(device)

            outputs = model(input_image)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")


def evaluate(criterion, device, model, optimizer, dataloader, transform, measurer):
    model.eval()
    all_predictions = []
    all_labels = []
    running_loss = 0.0

    for batch_id, (images, mag1c, labels) in enumerate(dataloader):
        input_image = torch.cat((images, mag1c), dim=1)
        input_image = transform(input_image).to(device)
        labels = labels.long().to(device)

        outputs = model(input_image)
        predictions = torch.argmax(outputs, dim=1)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        all_predictions.append(predictions.cpu().detach())
        all_labels.append(labels.cpu().detach())

    measures = measurer.compute_measures(torch.cat(all_predictions), torch.cat(all_labels))
    print(f"Validation loss: {running_loss / len(dataloader)}.\nMeasures: {measures}")
    return measures


if __name__ == "__main__":
    epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 1e-4

    train_dataloader, test_dataloader = setup_dataloaders()
    model, criterion, optimizer = setup_model(lr, device)
    model_handler = ModelFilesHandler()
    measurer = MeasureToolFactory.get_measure_tool(ModelType.TRANSFORMER)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.Normalize(mean=[0.5] * 9, std=[0.5] * 9)  # Normalize for 9 channels
    ])

    train(criterion, device, epochs, model, optimizer, train_dataloader, transform)
    measures = evaluate(criterion, device, model, optimizer, test_dataloader, transform, measurer)

    model_handler.save_model(
        model=model,
        metrics = measures,
        model_type=ModelType.TRANSFORMER_CLASSIFIER,
        epoch=epochs,
    )
