import torch

from torch.utils.data import DataLoader
from transformers import AdamW

from dataset.STARCOP_dataset import STARCOPDataset
from dataset.dataset_info import SegmentationDatasetInfo
from dataset.dataset_type import DatasetType
from models.Transformer.DETR.model import CustomDetr

from models.Tools.FilesHandler.model_files_handler import ModelFilesHandler
from models.Tools.Measures.measure_tool_factory import MeasureToolFactory
from models.Tools.Measures.model_type import ModelType


def create_datasets():
    dataset_train = STARCOPDataset(
        data_path=r"data",
        data_type=DatasetType.EASY_TRAIN,
        image_info_class=SegmentationDatasetInfo,
        normalization=False
    )

    dataset_test = STARCOPDataset(
        data_path=r"data",
        data_type=DatasetType.TEST,
        image_info_class=SegmentationDatasetInfo,
        normalization=False
    )

    return dataset_train, dataset_test

if __name__ == "__main__":
    num_epochs = 50
    learning_rate = 1e-5
    batch_size = 64
    device = "cuda"

    model = CustomDetr()
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    measurer = MeasureToolFactory.get_measure_tool(ModelType.TRANSFORMER)

    dataset_train, dataset_test = create_datasets()

    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    measures = None
    for epoch in range(num_epochs):
        total_loss = 0

        all_target_masks = []
        all_pred_masks = []
        for batch_idx, (image, _, mag1c, _, mask, bboxes, labels) in enumerate(train_dataloader):
            image = image.to(device)
            mag1c = mag1c.to(device)
            input_data = torch.cat((image, mag1c), dim=1)

            target_bboxes = bboxes.to(device)
            target_labels = labels.to(device)
            target_mask = mask.to(device)

            targets = [
                {"boxes": target_bboxes[i], "class_labels": target_labels[i], "masks": target_mask[i]}
                for i in range(target_bboxes.size(0))
            ]

            optimizer.zero_grad()

            outputs = model(pixel_values=input_data, labels=targets)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            output_masks = outputs.pred_masks
            output_masks_upsampled = torch.nn.functional.interpolate(output_masks, size=(256, 256), mode='bilinear', align_corners=False)
            all_target_masks.append(target_mask)
            all_pred_masks.append(output_masks_upsampled)

            print(f"Batch number: {batch_idx}, Loss: {loss}")



        measures = measurer.compute_measures(torch.cat(all_pred_masks, dim=0), torch.cat(all_target_masks, dim=0))
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader)}")



    model_handler = ModelFilesHandler()
    model_handler.save_model(
        model=model,
        epoch=10,
        metrics=measures,
        model_type=ModelType.TRANSFORMER
    )