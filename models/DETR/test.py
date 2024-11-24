import cv2
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from dataset.STARCOP_dataset import STARCOPDataset
from dataset.dataset_info import FilteredSpectralImageInfo
from dataset.dataset_type import DatasetType
from models.Tools.model_files_handler import ModelFilesHandler
from models.DETR.model import DetrWith9Channels

if __name__ == '__main__':
    # Load original image

    # Create dataset
    dataset = STARCOPDataset(
        data_path=r"data",
        data_type=DatasetType.EASY_TRAIN,
        image_info_class=FilteredSpectralImageInfo,
        normalization=False
    )

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Load single image
    image_result = None
    for image in dataloader:
        mask =  image[5]
        if not torch.all(mask == 0):
            image_result = image
            break
    image = image_result

    # Load model
    model_handler = ModelFilesHandler()
    model, _, _, _ = model_handler.load_model(file_name=r"trained_models\model_transformer_2024_11_24_22_45_54.pickle")
    model.eval()
    model.to("cuda")

    mask = image[4][0, 0, :, :]

    targets = [{
        "boxes": image[5].squeeze(0).to("cuda"),
        "class_labels": image[6].squeeze(0).to("cuda")
     }]
    outputs = model(torch.cat((image[0], image[2]), dim=1).to("cuda"), labels=targets)

    print(outputs.pred_boxes)
    print(outputs.logits)

    predicted_bbox = outputs.pred_boxes
    predicted_labels = outputs.logits

    # Background
    # background = image[0][0, :3, :, :].permute(1, 2, 0).numpy()
    # print(background.shape)

    background = mask.unsqueeze(2).expand(-1, -1, 3).numpy()


    plt.title=""
    canvas = background.copy()  # Avoid modifying the original mask


    for index, (bbox, confidence) in enumerate(zip(predicted_bbox.squeeze(0), predicted_labels.squeeze(0))):
        bbox = (bbox * 256).int()

        cx = bbox[0].item()
        cy = bbox[1].item()
        w = bbox[2].item()
        h = bbox[3].item()

        pt1 = (cx - w // 2, cy - h // 2)  # Top-left corner
        pt2 = (cx + w // 2, cy + h // 2)  # Bottom-right corner

        # Draw rectangle on the binary mask
        cv2.rectangle(
            canvas,
            pt1=pt1,
            pt2=pt2,
            color=(1, 0, 0),
            thickness=1
        )


    for index, (bbox, confidence) in enumerate(zip(image[5].squeeze(0), image[6].squeeze(0))):
        # Stop processing if confidence is negative
        # Scale and convert bounding box to integer coordinates
        bbox = (bbox * 256).int()

        cx = bbox[0].item()
        cy = bbox[1].item()
        w  = bbox[2].item()
        h  = bbox[3].item()

        pt1 = (cx - w // 2, cy - h // 2)  # Top-left corner
        pt2 = (cx + w // 2, cy + h // 2)  # Bottom-right corner

        # Draw rectangle on the binary mask
        cv2.rectangle(
            canvas,
            pt1=pt1,
            pt2=pt2,
            color=(0, 1, 0),
            thickness=1
        )

    # for index, (bbox, confidence) in enumerate(zip(predicted_bbox.squeeze(0).sigmoid(), predicted_labels.squeeze(0))):
    #     bbox = (bbox * 256).int()
    #
    #     cx = bbox[0].item()
    #     cy = bbox[1].item()
    #     w = bbox[2].item()
    #     h = bbox[3].item()
    #
    #     pt1 = (cx - w / 2, cy - h / 2)  # Top-left corner
    #     pt2 = (cx + w / 2, cy + h / 2)  # Bottom-right corner
    #
    #     # Draw rectangle on the binary mask
    #     cv2.rectangle(
    #         canvas,
    #         pt1=pt1,
    #         pt2=pt2,
    #         color=(0, 0, 1),
    #         thickness=1
    #     )

    plt.imshow(canvas,)
    plt.show()
