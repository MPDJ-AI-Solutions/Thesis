import torch

from models.Tools.FilesHandler.model_files_handler import ModelFilesHandler
from models.Tools.Measures.measure_tool_factory import MeasureToolFactory
from models.Tools.Measures.model_type import ModelType
from models.Tools.Train.train_classifier import setup_dataloaders, setup_model, train, evaluate
from models.Transformer.VIT.model import CustomViT

if __name__ == "__main__":
    epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 1e-4

    train_dataloader, test_dataloader = setup_dataloaders()
    model = CustomViT()
    model, criterion, optimizer = setup_model(model, lr, device)
    model_handler = ModelFilesHandler()
    measurer = MeasureToolFactory.get_measure_tool(ModelType.TRANSFORMER)


    train(criterion, device, epochs, model, optimizer, train_dataloader,model_handler, log_batches=True)
    measures = evaluate(criterion, device, model, test_dataloader, measurer)

    model_handler.save_model(
        model=model,
        metrics = measures,
        model_type=ModelType.TRANSFORMER_CLASSIFIER,
        epoch=epochs,
    )
