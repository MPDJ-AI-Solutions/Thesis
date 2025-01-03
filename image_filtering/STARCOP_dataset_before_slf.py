import os
import pandas as pd

from pathlib import Path
from typing import Type
from torch.utils.data import Dataset

from dataset.dataset_info import DatasetInfo
from dataset.dataset_type import DatasetType


class STARCOPDatasetPreSLF(Dataset):
    """
    Class is used for custom dataloader. Loads images based on CSV description of data.
    """
    def __init__(self, data_path: str, data_type: DatasetType, image_info_class: Type[DatasetInfo]):
        """
        Initializes the dataset processing class.
        Args:
            data_path (str): The path to the dataset directory.
            data_type (DatasetType): The type of dataset being processed.
            image_info_class (Type[DatasetInfo]): The class containing image information.
        Attributes:
            images_path (str): The path to the images directory.
            image_info (Type[DatasetInfo]): The class containing image information.
            csv (pd.DataFrame): The DataFrame containing dataset information, filtered to exclude already processed files.
        Prints:
            The number of files left to process.
        """
        self.images_path = os.path.join(data_path, data_type.get_folder_name(), data_type.get_folder_name())
        self.image_info = image_info_class
        self.csv = pd.read_csv(os.path.join(data_path, data_type.value + ".csv"))

        self.csv = self.csv[self.csv["id"].apply(lambda img_id: not self._processed_file_exists(img_id))].reset_index(drop=True)
        print(f"{len(self.csv)} files to process left.")

    def _processed_file_exists(self, img_id):
        """
        Check if the processed file 'slf_result.npy' exists for a given image ID.

        Args:
            img_id (str): The identifier of the image.

        Returns:
            bool: True if the processed file exists, False otherwise.
        """
        processed_file_path = Path(self.images_path) / img_id / "slf_result_new1.npy"
        return processed_file_path.exists()

    def __len__(self):
        """
        Returns the number of entries that are not processed yet in the dataset.

        Returns:
            int: The number of entries in the CSV file.
        """
        return len(self.csv)

    def __getitem__(self, index):
        """
        Retrieves the image tensor and its directory path for the given index.

        Args:
            index (int): The index of the image to retrieve.

        Returns:
            tuple: A tuple containing the image tensor and the directory path of the image.
        """
        images_directory_path = os.path.join(self.images_path, self.csv["id"][index])
        return self.image_info.load_tensor(images_directory_path)[0], images_directory_path
