import torch

from dataset.dataset_type import DatasetType

from models.Convolutional.MethaNet.model import MethaNetClassifier
from files_handler import ModelFilesHandler
from measures import MeasureToolFactory
from measures import ModelType
from models.Tools.Train.train_classifier import setup_dataloaders, setup_model, train, evaluate


if __name__ == "__main__":
    epochs = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 1e-4

    train_dataloader, test_dataloader = setup_dataloaders(batch_size=32, train_type=DatasetType.TRAIN)
    model = MethaNetClassifier()
    model, criterion, optimizer = setup_model(model, lr, device)

    model_handler = ModelFilesHandler()
    measurer = MeasureToolFactory.get_measure_tool(ModelType.CNN)

    train(criterion, device, epochs, model, optimizer, train_dataloader, model_handler, log_batches=True)
    measures = evaluate(criterion, device, model, test_dataloader, measurer)

    model_handler.save_model(
        model=model,
        metrics=measures,
        model_type=ModelType.CNN,
        epoch=epochs,
    )