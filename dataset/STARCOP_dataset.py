import os
import pandas as pd
import torch

from torch.utils.data import Dataset

from .dataset_info import SpectralImageInfo
from .dataset_type import DatasetType
from typing import Type


class STARCOPDataset(Dataset):
    """
    Class is used for custom dataloader. Loads images based on CSV description of data.
    """

    def __init__(
            self,
            data_path: str,
            data_type: DatasetType,
            image_info_class: Type[SpectralImageInfo],
            enable_augmentation: bool = False
    ):
        self.images_path = os.path.join(data_path, data_type.get_folder_name(), data_type.get_folder_name())
        self.image_info = image_info_class
        self.csv = pd.read_csv(os.path.join(data_path, data_type.value + ".csv"))
        self.csv = self.csv[self.csv['has_plume'] == True].reset_index(drop=True)
        self.enable_augmentation = enable_augmentation

    def __len__(self):
        return len(self.csv) * 2 if self.enable_augmentation else len(self.csv)

    def __getitem__(self, index):
        return self._augmented_get_item(index) if self.enable_augmentation else self._normal_get_item(index)

    def _augmented_get_item(self, index):
        csv_index = int(index / 2)
        image_directory_path = os.path.join(self.images_path, self.csv["id"][csv_index])
        result_image = self.image_info.load_tensor(image_directory_path)
        image, filtered_image, _, _, mask, _ = result_image
        if index % 2 == 0:
            return result_image
        else:
            image, filtered_image, mask = STARCOPDataset._augment(image, filtered_image, mask)
            return image, filtered_image, result_image[2], result_image[3], mask, result_image[5]

    def _normal_get_item(self, index):
        images_directory_path = os.path.join(self.images_path, self.csv["id"][index])

        return self.image_info.load_tensor(images_directory_path)

    @staticmethod
    def _augment(image, filtered_image, mask):
        """
        Augment a hyperspectral image using various techniques.
        Args:
            image (torch.Tensor): Input image of shape (C, H, W).
        """
        _, H, W = image.shape

        # Horizontal Flip
        if torch.rand(1).item() > 0.5:
            image = image.flip(-1)
            filtered_image = filtered_image.flip(-1)
            mask = mask.flip(-1)

        # Rotation (90° increments)
        rotations = torch.randint(0, 4, (1,)).item()  # 0, 1, 2, or 3
        image = image.rot90(rotations, [1, 2])
        filtered_image = filtered_image.rot90(rotations, [1, 2])
        mask = mask.rot90(rotations, [1, 2])

        return image, filtered_image, mask
