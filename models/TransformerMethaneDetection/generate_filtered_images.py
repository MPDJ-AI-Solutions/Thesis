import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from dataset.STARCOP_dataset_before_slf import STARCOPDatasetPreSLF
from dataset.dataset_info import TransformerModelSpectralImageInfo
from dataset.dataset_type import DatasetType
from models.TransformerMethaneDetection.SpectralFeatureGenerator.spectral_linear_filter import SpectralLinearFilterParallel

# TODO find place for this file (e.g. create module for generating this)

if __name__ == '__main__':
    methane_pattern = [0, 0, 0, 0.1, 0.3, 0.6, 0.8, 0.7]

    dataset = STARCOPDatasetPreSLF(
        r"../../data",
        data_type=DatasetType.TRAIN,
        image_info_class=TransformerModelSpectralImageInfo
    )
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True)


    model = SpectralLinearFilterParallel(num_classes=20, min_class_size=10000)

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, file_paths) in enumerate(dataloader):
            images, _ = TransformerModelSpectralImageInfo.backbone_input_converter(data)
            processed_images = model(images, methane_pattern)
            for img, file_path in zip(processed_images, file_paths):
                save_path = Path(os.path.join(file_path, Path(file_path).stem)).with_name(f"slf_result.npy")

                img_np = img.cpu().numpy()
                np.save(save_path, img_np)
                print(f"Batch {batch_idx} saved at {save_path}.")
