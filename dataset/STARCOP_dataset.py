import os
import pandas as pd
import torch
import torchvision

from torch.utils.data import Dataset, DataLoader

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
            normalization: bool = False
    ):
        self.images_path = os.path.join(data_path, data_type.get_folder_name(), data_type.get_folder_name())
        self.image_info = image_info_class
        self.csv = pd.read_csv(os.path.join(data_path, data_type.value + ".csv"))
        self.csv = self.csv[self.csv['has_plume'] == True].reset_index(drop=True)
        self.normalization = normalization

        self.image_mean = 0
        self.image_std = 0
        self.mag1c_mean = 0
        self.mag1c_std = 0

        if normalization:
            self.normalization = False
            self.image_mean, self.image_std = self._calculate_image_stats()
            self.mag1c_mean, self.mag1c_std = self._calculate_mag1c_stats()

            # self.image_mean = torch.tensor([3.3703e-06, 3.6121e-05, 2.1833e-05, 1.2279e-05, 1.5941e-05, 3.4616e-04, 3.7622e-04, 4.0710e-04])
            # self.image_std = torch.tensor([1.7995e-06, 1.4989e-05, 9.2557e-06, 5.3556e-06, 6.8728e-06, 1.3327e-04, 1.5923e-04, 1.8511e-04])
            # self.mag1c_mean = torch.tensor([0.0007])
            # self.mag1c_std = torch.tensor([0.0033])

            print(50*'-')
            print(f"Dataset: {data_type.name}")
            print(f"Image mean: {self.image_mean}, Image std: {self.image_std}")
            print(f"Mag1c mean: {self.mag1c_mean}, Mag1c std: {self.mag1c_std}")
            print(50*'-')
            self.normalization = True

        self.normalize_image = torchvision.transforms.Normalize(mean=self.image_mean, std=self.image_std)
        self.normalize_mag1c = torchvision.transforms.Normalize(mean=self.mag1c_mean, std=self.mag1c_std)

    def __len__(self):
        return len(self.csv) * 4

    def __getitem__(self, index):
        return self._get_normalized_item(index) if self.normalization else self._get_raw_item(index)

    def _get_normalized_item(self, index):
        images = self._get_raw_item(index)
        normalized_image = self.normalize_image(images[0])
        normalized_mag1c = self.normalize_mag1c(images[2])

        results = normalized_image, images[1], normalized_mag1c, images[3], images[4], images[5], images[6]
        return results

    def _get_raw_item(self, index):
        file_index = index // 4
        images_directory_path = os.path.join(self.images_path, self.csv["id"][file_index])

        return self.image_info.load_tensor(images_directory_path, grid_id=index % 4)

    def _calculate_image_stats(self) -> (float, float):
        dataloader = DataLoader(self, batch_size=16, shuffle=False)

        n_channels = 8
        mean = torch.zeros(n_channels)
        std = torch.zeros(n_channels)
        num_pixels = 0

        for images, *_ in dataloader:  # Unpack only images
            images = images.view(images.size(0), images.size(1), -1)  # Flatten HxW
            mean += images.mean(dim=2).sum(dim=0)
            std += images.std(dim=2).sum(dim=0)
            num_pixels += images.size(0) * images.size(2)

        mean /= num_pixels
        std /= num_pixels
        return mean, std

    def _calculate_mag1c_stats(self) -> (float, float):
        dataloader = DataLoader(self, batch_size=16, shuffle=False)

        n_channels = 1
        mean = torch.zeros(n_channels)
        std = torch.zeros(n_channels)
        num_pixels = 0

        for _, _, mag1c, *_ in dataloader:  # Unpack only images
            mag1c = mag1c.view(mag1c.size(0), mag1c.size(1), -1)  # Flatten HxW
            mean += mag1c.mean(dim=2).sum(dim=0)
            std += mag1c.std(dim=2).sum(dim=0)
            num_pixels += mag1c.size(0) * mag1c.size(2)

        mean /= num_pixels
        std /= num_pixels
        return mean, std
