import torch

from dataset.dataset_info import MMClassifierDatasetInfo
from dataset.dataset_type import DatasetType
from files_handler.model_files_handler import ModelFilesHandler
from measures import MeasureToolFactory
from measures import ModelType
from models.Tools.Train.train_classifier import setup_model, setup_dataloaders, \
    print_progress_bar
from models.Transformer.MethaneMapper.model import TransformerModel


def train(criterion, device, epochs, model, optimizer, dataloader, model_handler,  log_batches: bool = False):
    """
    Trains a given model using the specified criterion, optimizer, and dataloader. Specified for MethaneMapper
    
    Args:
        criterion (torch.nn.Module): The loss function to be used.
    device (torch.device): The device to run the training on (e.g., 'cpu' or 'cuda').
        epochs (int): The number of epochs to train the model for.
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer to use for updating the model parameters.
        dataloader (torch.utils.data.DataLoader): The DataLoader providing the training data.
        model_handler (object): An object responsible for saving the model.
        log_batches (bool, optional): If True, logs progress every 10 batches. Defaults to False.
    Returns:
        None
    """
    model.train()
    for epoch in range(epochs):  # Adjust the number of epochs
        running_loss = 0.0
        for batch_id, (images, mag1c, filtered_image, labels) in enumerate(dataloader):  # Assume a PyTorch DataLoader is set up
            optimizer.zero_grad()

            input_image = torch.cat((images, mag1c), dim=1).to(device)
            filtered_image = filtered_image.to(device)
            labels = labels.long().to(device)

            # Change for MethaneMapper
            outputs = model(input_image, filtered_image)

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
    Evaluates the performance of a model on a given dataset for MethaneMapper model.
    
    Args:
        criterion (torch.nn.Module): The loss function used to evaluate the model.
        device (torch.device): The device (CPU or GPU) on which the model and data are located.
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

    for batch_id, (images, mag1c, filtered_image, labels) in enumerate(dataloader):
        input_image = torch.cat((images, mag1c), dim=1).to(device)
        filtered_image = filtered_image.to(device)
        labels = labels.long().to(device)

        # Change for MethaneMapper
        outputs = model(input_image, filtered_image)

        predictions = torch.argmax(outputs, dim=1)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        all_predictions.append(predictions.cpu().detach())
        all_labels.append(labels.cpu().detach())

    measures = measurer.compute_measures(torch.cat(all_predictions), torch.cat(all_labels))
    print(f"Validation loss: {running_loss / len(dataloader)}.\nMeasures:\n{measures}")
    return measures


if __name__ == "__main__":
    epochs = 15
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 1e-4

    train_dataloader, test_dataloader = setup_dataloaders(
        batch_size=4,
        image_info_class=MMClassifierDatasetInfo,
        crop_size=1,
        train_type=DatasetType.TRAIN,
    )
    model = TransformerModel(
        n_queries=5,
        n_decoder_layers=5,
        n_encoder_layers=5,
        d_model=256,
    )
    model, criterion, optimizer = setup_model(model, lr, device)
    model_handler = ModelFilesHandler()
    measurer = MeasureToolFactory.get_measure_tool(ModelType.TRANSFORMER)

    # Training
    print("Training...")
    train(criterion, device, epochs, model, optimizer, train_dataloader, model_handler, log_batches=True)

    model_handler.save_raw_model(model)

    # Validation
    print("Evaluating...")
    measures = evaluate(criterion, device, model, test_dataloader, measurer)

    model_handler.save_model(
        model=model,
        metrics=measures,
        model_type=ModelType.TRANSFORMER_CLASSIFIER,
        epoch=epochs,
    )

# if __name__ == "__main__":
#     epochs = 15
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     lr = 1e-4
#
#     train_dataloader, test_dataloader = setup_dataloaders(
#         batch_size=2,
#         image_info_class=MMClassifierDatasetInfo,
#         crop_size=1,
#         train_type=DatasetType.TRAIN,
#     )
#
#     model_handler = ModelFilesHandler()
#     model, _, _, _ = model_handler.load_model(file_name=r"trained_models/model_transformer_classifier_2024_12_02_00_37_54.pickle")
#     model, criterion, optimizer = setup_model(model, lr, device)
#     measurer = MeasureToolFactory.get_measure_tool(ModelType.TRANSFORMER)
#
#     # Validation
#     print("Evaluating...")
#     measures = evaluate(criterion, device, model, test_dataloader, measurer)
#
#     model_handler.save_model(
#         model=model,
#         metrics=measures,
#         model_type=ModelType.TRANSFORMER_CLASSIFIER,
#         epoch=epochs,
#     )

