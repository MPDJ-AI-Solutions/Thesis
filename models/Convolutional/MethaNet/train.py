import torch

from dataset.STARCOP_dataset import STARCOPDataset
from dataset.dataset_info import SegmentationDatasetInfo
from dataset.dataset_type import DatasetType


from models.Convolutional.MethaNet.model import MethaNetClassifier
from models.Tools.FilesHandler.model_files_handler import ModelFilesHandler
from models.Tools.Measures.measure_tool_factory import MeasureToolFactory
from models.Tools.Measures.model_type import ModelType
from models.Tools.Train.train_classifier import setup_dataloaders, setup_model, train_cnn, evaluate_cnn, train, evaluate

def create_datasets():
    dataset_train = STARCOPDataset(
        data_path=r"data",
        data_type=DatasetType.EASY_TRAIN,
        image_info_class=SegmentationDatasetInfo,
        normalization=False
    )

    dataset_test = STARCOPDataset(
        data_path=r"data",
        data_type=DatasetType.EASY_TRAIN,
        image_info_class=SegmentationDatasetInfo,
        normalization=False
    )

    return dataset_train, dataset_test

# 50/100/200 epoch
# images/images+mag1c
# EASY_TRAIN/TRAIN

############### TESTING ###############
# epoch | images/images+magic | dataset | learning_rate

# WITH PRE_CONV
# 100 | images | EASY_TRAIN | -4
# 100 | images | TRAIN | -4
# 100 | images+mag1c | TRAIN | -4
# 50 | images+mag1c | EASY_TRAIN | -4
# 200 | images+mag1c | TRAIN | -5

# NO PRE_CONV
# 100 | images | EASY_TRAIN | -4


if __name__ == "__main__":
    epochs = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 1e-4

    train_dataloader, test_dataloader = setup_dataloaders(batch_size=80)
    model = MethaNetClassifier()
    model, criterion, optimizer = setup_model(model, lr, device)

    model_handler = ModelFilesHandler()
    measurer = MeasureToolFactory.get_measure_tool(ModelType.CNN)

    train_cnn(criterion, device, epochs, model, optimizer, train_dataloader, log_batches=True)
    measures = evaluate_cnn(criterion, device, model, test_dataloader, measurer)

    model_handler.save_model(
        model=model,
        metrics=measures,
        model_type=ModelType.CNN,
        epoch=epochs,
        )