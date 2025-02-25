import os
import torch
import torchvision
import pandas as pd

from torch.utils.data import Dataset, DataLoader

from .dataset_info import DatasetInfo
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
            image_info_class: Type[DatasetInfo],
            crop_size: int = 1,
            normalization: bool = False
    ):
        """
            Initializes the dataset object.

            Args:
                data_path (str): The path to the dataset directory.
                data_type (DatasetType): The type of dataset being used.
                image_info_class (Type[DatasetInfo]): The class containing image information.
                crop_size (int, optional): The size of the crop to be applied to images. Defaults to 1.
                normalization (bool, optional): Whether to apply normalization to the images. Defaults to False.
            """
        self.images_path = os.path.join(data_path, data_type.get_folder_name(), data_type.get_folder_name())
        self.image_info = image_info_class
        self.csv = pd.read_csv(os.path.join(data_path, data_type.value + ".csv"))
        self.normalization = normalization
        self.crop_size = crop_size

        self.image_mean = 0
        self.image_std = 0
        self.mag1c_mean = 0
        self.mag1c_std = 0

        if normalization:
            self.normalization = False
            self.image_mean, self.image_std = self._calculate_image_stats()
            self.mag1c_mean, self.mag1c_std = self._calculate_mag1c_stats()
            self.normalization = True
            self.print_std_mean(data_type, self.image_mean, self.image_std, self.mag1c_mean, self.mag1c_std)

        self.normalize_image = torchvision.transforms.Normalize(mean=self.image_mean, std=self.image_std)
        self.normalize_mag1c = torchvision.transforms.Normalize(mean=self.mag1c_mean, std=self.mag1c_std)

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        The total number is calculated as the length of the CSV file
        multiplied by the square of the crop size.

        Returns:
            int: The total number of cropped images.
        """
        return len(self.csv) * (self.crop_size * self.crop_size)

    def __getitem__(self, index):
        """
        Retrieve an item from the dataset at the specified index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            The normalized item if normalization is enabled, otherwise the raw item.
        """
        return self._get_normalized_item(index) if self.normalization else self._get_raw_item(index)

    def _get_raw_item(self, index):
        """
        Retrieves a raw image tensor from the dataset based on the given index.

        Args:
            index (int): The index of the image to retrieve.
        Returns:
            tuple: Hyperspectral image loaded via ImageInfoClass.
        """
        file_index = index // (self.crop_size * self.crop_size)
        images_directory_path = os.path.join(self.images_path, self.csv["id"][file_index])

        return self.image_info.load_tensor(
            images_directory_path,
            grid_id=index % (self.crop_size * self.crop_size),
            crop_size=self.crop_size
        )

    def _get_normalized_item(self, index):
        """
        Retrieves and normalizes a dataset item by its index.

        Args:
            index (int): The index of the item to retrieve.
        Returns:
            tuple: A tuple containing the normalized image. 
        """
        images = self._get_raw_item(index)
        normalized_image = self.normalize_image(images[0])
        normalized_mag1c = self.normalize_mag1c(images[2])

        results = normalized_image, images[1], normalized_mag1c, images[3], images[4], images[5], images[6]
        return results

    def _calculate_image_stats(self) -> (float, float):
        """
        Calculates statistics for hyperspectral image. 

        Returns:
            tuple: A tuple containing mean and standard deviation for hsi.
        """
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
        """
        Calculates statistics for mag1c image. 

        Returns:
            tuple: A tuple containing mean and standard deviation for mag1c.
        """
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

    def print_std_mean(self, data_type):
        """
        Prints the mean and standard deviation of the dataset for the specified data type.

        Args:
            data_type (Enum): The type of data for which the statistics are being printed. 
                              It should have a 'name' attribute that will be used in the print statement.
        """
        print(100 * '-')
        print(f"Dataset: {data_type.name}")
        print(f"Image mean: {self.image_mean}, Image std: {self.image_std}")
        print(f"Mag1c mean: {self.mag1c_mean}, Mag1c std: {self.mag1c_std}")
        print(100 * '-')
