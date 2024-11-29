import torch
from torchvision import transforms

from models.Transformer.DETR.model import CustomDetrForClassification
from models.Tools.FilesHandler.model_files_handler import ModelFilesHandler
from models.Tools.Measures.measure_tool_factory import MeasureToolFactory
from models.Tools.Measures.model_type import ModelType
from models.Tools.Train.train_classifier import setup_dataloaders, setup_model, train, evaluate


if __name__ == "__main__":
    epochs = 20
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 1e-5

    train_dataloader, test_dataloader = setup_dataloaders()
    model = CustomDetrForClassification()
    model, criterion, optimizer = setup_model(model, lr, device)
    model_handler = ModelFilesHandler()
    measurer = MeasureToolFactory.get_measure_tool(ModelType.TRANSFORMER)

    transform = transforms.Compose([transforms.ToTensor(), ])

    train(criterion, device, epochs, model, optimizer, train_dataloader, transform)
    measures = evaluate(criterion, device, model, test_dataloader, transform, measurer)

    model_handler.save_model(
        model=model,
        metrics=measures,
        model_type=ModelType.TRANSFORMER_CLASSIFIER,
        epoch=epochs,
    )
