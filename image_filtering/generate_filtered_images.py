import os
import numpy as np
import torch

from pathlib import Path
from torch.utils.data import DataLoader

from dataset.dataset_info import SegmentationDatasetInfo
from dataset.dataset_type import DatasetType
from image_filtering.STARCOP_dataset_before_slf import STARCOPDatasetPreSLF
from models.Transformer.MethaneMapper.SpectralFeatureGenerator.spectral_linear_filter import \
    SpectralLinearFilterParallel


def generate_images(data_type):
    """
    Generates filtered images using a spectral linear filter and saves the results.
     The function performs the following steps:
    1. Initializes a dataset of type `STARCOPDatasetPreSLF` with the given `data_type`.
    2. Creates a DataLoader to iterate over the dataset in batches.
    3. Initializes a `SpectralLinearFilterParallel` model for image filtering.
    4. Sets the model to evaluation mode and processes each batch of images without computing gradients.
    5. Applies the spectral linear filter to each image in the batch using a predefined methane pattern.
    6. Saves the processed images as `.npy` files in the same directory as the original images, with filenames indicating the result of the spectral linear filter.
    
    Args:
        data_type (str): The type of data to be processed, used to initialize the dataset.
    """
    methane_pattern = [0.1, 0.3, 0.6, 0.8, 0.7, 0, 0, 0]

    dataset = STARCOPDatasetPreSLF(
        r"data",
        data_type=data_type,
        image_info_class=SegmentationDatasetInfo
    )
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True)

    model = SpectralLinearFilterParallel(num_classes=20, min_class_size=10000)

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, file_paths) in enumerate(dataloader):
            processed_images = model(images, methane_pattern)
            for img, file_path in zip(processed_images, file_paths):
                save_path = Path(os.path.join(file_path, Path(file_path).stem)).with_name(f"slf_result_new1.npy")

                img_np = img.cpu().numpy()
                np.save(save_path, img_np)
                print(f"Batch {batch_idx} saved at {save_path}.")


if __name__ == '__main__':
    generate_images(data_type=DatasetType.TEST)
