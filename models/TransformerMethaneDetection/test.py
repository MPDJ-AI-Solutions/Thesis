import cv2
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from dataset.STARCOP_dataset import STARCOPDataset
from dataset.dataset_info import FilteredSpectralImageInfo
from dataset.dataset_type import DatasetType
from models.Tools.model_files_handler import ModelFilesHandler

if __name__ == '__main__':
    # Load original image

    # Create dataset
    dataset = STARCOPDataset(
        data_path=r"data",
        data_type=DatasetType.EASY_TRAIN,
        image_info_class=FilteredSpectralImageInfo,
        enable_augmentation=False
    )

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Load single image
    image = dataloader.__iter__().__next__()

    # Load model
    model_handler = ModelFilesHandler()
    model, _, _, _ = model_handler.load_model(file_name=r"trained_models\model_transformer_2024_11_17_22_13_18.pickle")

    mask = image[4][0, 0, :, :]
    model.to("cpu")
    predicted_bbox, predicted_confidence, predicted_mask = model(image[0], image[1])

    plt.title="Original mask"
    plt.imshow(mask, cmap='gray')
    plt.show()

    plt.title="Original mask with bbox"

    binary_mask = mask.squeeze(0).expand(3, 512, 512).permute(1, 2, 0).numpy()
    binary_mask_copy = binary_mask.copy()  # Avoid modifying the original mask

    for index, (bbox, confidence) in enumerate(zip(image[5].squeeze(0), image[6].squeeze(0))):
        # Stop processing if confidence is negative
        if confidence < 1:
            break

        # Scale and convert bounding box to integer coordinates
        bbox = bbox.int()

        pt1 = (bbox[0].item(), bbox[1].item())  # Top-left corner
        pt2 = (bbox[0].item() + bbox[2].item(), bbox[1].item() + bbox[3].item())  # Bottom-right corner

        # Draw rectangle on the binary mask
        cv2.rectangle(
            binary_mask_copy,
            pt1=pt1,
            pt2=pt2,
            color=(1, 0, 0),
            thickness=1
        )
    plt.imshow(binary_mask_copy, cmap='gray')
    plt.show()

    plt.title="Computed mask"
    plt.imshow(predicted_mask.detach().squeeze(0).permute(1, 2, 0), cmap='gray')
    plt.show()

    plt.title="Computed mask with bbox"
    predicted_mask = (predicted_mask > 0.5).float()
    binary_mask = predicted_mask.squeeze(0).expand(3, 512, 512).permute(1, 2, 0).detach().numpy()
    binary_mask_copy = binary_mask.copy()  # Avoid modifying the original mask

    canvas = torch.zeros(512, 512, 1).numpy()

    print(predicted_confidence)
    for index, (bbox, confidence) in enumerate(zip(predicted_bbox.squeeze(0), predicted_confidence.squeeze(0))):
        # Stop processing if confidence is negative

        bbox = bbox.int()
        # Scale and convert bounding box to integer coordinates
        pt1 = (bbox[0].item(), bbox[1].item())  # Top-left corner
        pt2 = (bbox[0].item() + bbox[2].item(), bbox[1].item() + bbox[3].item())  # Bottom-right corner

        # Draw rectangle on the binary mask
        cv2.rectangle(
            canvas,
            pt1=pt1,
            pt2=pt2,
            color=(255, 255, 0),
            thickness=0
        )
    plt.imshow(canvas, cmap='gray')
    plt.show()

