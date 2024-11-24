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
    image_result = None
    for image in dataloader:
        mask =  image[5]
        if not torch.all(mask == 0):
            image_result = image
            break

    # Load model
    image = image_result
    model_handler = ModelFilesHandler()
    model, _, _, _ = model_handler.load_model(file_name=r"trained_models\model_transformer_2024_11_24_13_30_30.pickle")
    mask = image[4][0, 0, :, :]
    model.to("cuda")
    predicted_bbox, predicted_confidence = model(image[0].to("cuda"), image[1].to("cuda"), image[2].to("cuda"))

    print(image[5])
    print(predicted_bbox)
    mag1c = image[0][0, :3, :, :].permute(1, 2, 0).numpy()
    tensor_min = 0
    tensor_max = 30

    mag1c = (mag1c - tensor_min) / (tensor_max - tensor_min)
    print(mag1c.shape)
    # plt.title = "Mapa"
    # plt.imshow(mag1c, cmap='gray')
    # plt.show()

    plt.title="Original mask"
    #plt.imshow(mask, cmap='gray')
    plt.show()

    plt.title="Original mask with bbox"

    binary_mask = mag1c
    binary_mask_copy = mag1c.copy()  # Avoid modifying the original mask

    for index, (bbox, confidence) in enumerate(zip(image[5].squeeze(0), image[6].squeeze(0))):
        # Stop processing if confidence is negative
        # Scale and convert bounding box to integer coordinates
        if confidence < 1:
            continue
        bbox = (bbox * 128).int()

        pt1 = (bbox[0].item(), bbox[1].item())  # Top-left corner
        pt2 = (bbox[2].item(), bbox[3].item())  # Bottom-right corner

        # Draw rectangle on the binary mask
        cv2.rectangle(
            binary_mask_copy,
            pt1=pt1,
            pt2=pt2,
            color=(0, 1, 0),
            thickness=1
        )
    #plt.imshow(binary_mask_copy, cmap='gray')
    plt.show()

    plt.title="Computed bbox"
    canvas = torch.zeros(512, 512, 1).numpy()

    print(predicted_confidence.sigmoid())
    for index, (bbox, confidence) in enumerate(zip(predicted_bbox.squeeze(0), predicted_confidence.squeeze(0))):
        # Stop processing if confidence is negative
        # if confidence[0] > confidence[1]:
        #     continue
        bbox = (bbox * 128).int()
        # Scale and convert bounding box to integer coordinates
        pt1 = (bbox[0].item(), bbox[1].item())  # Top-left corner
        pt2 = (bbox[2].item(), bbox[3].item())  # Bottom-right corner

        # Draw rectangle on the binary mask
        cv2.rectangle(
            binary_mask_copy,
            pt1=pt1,
            pt2=pt2,
            color=(1, 0, 0),
            thickness=0
        )
    plt.imshow(binary_mask_copy,)
    plt.show()

    for index, (bbox, confidence) in enumerate(zip(image[5].squeeze(0), image[6].squeeze(0))):
        # Stop processing if confidence is negative
        # Scale and convert bounding box to integer coordinates
        if confidence < 1:
            continue
        bbox = (bbox * 128).int()

        pt1 = (bbox[0].item(), bbox[1].item())  # Top-left corner
        pt2 = (bbox[2].item(), bbox[3].item())  # Bottom-right corner

        # Draw rectangle on the binary mask
        cv2.rectangle(
            binary_mask_copy,
            pt1=pt1,
            pt2=pt2,
            color=(0, 1, 0),
            thickness=1
        )
    #plt.imshow(binary_mask_copy, cmap='gray')
    plt.show()

    plt.title="Computed bbox"
    canvas = torch.zeros(512, 512, 1).numpy()

    print(predicted_confidence.sigmoid())
    for index, (bbox, confidence) in enumerate(zip(predicted_bbox.squeeze(0).sigmoid(), predicted_confidence.squeeze(0))):
        # Stop processing if confidence is negative
        # if confidence[0] > confidence[1]:
        #     continue
        bbox = (bbox * 128).int()
        # Scale and convert bounding box to integer coordinates
        pt1 = (bbox[0].item(), bbox[1].item())  # Top-left corner
        pt2 = (bbox[2].item(), bbox[3].item())  # Bottom-right corner

        # Draw rectangle on the binary mask
        cv2.rectangle(
            binary_mask_copy,
            pt1=pt1,
            pt2=pt2,
            color=(0, 0, 1),
            thickness=0
        )
    plt.imshow(binary_mask_copy,)
    plt.show()