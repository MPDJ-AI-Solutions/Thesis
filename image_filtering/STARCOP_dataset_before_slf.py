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
        self.images_path = os.path.join(data_path, data_type.get_folder_name(), data_type.get_folder_name())
        self.image_info = image_info_class
        self.csv = pd.read_csv(os.path.join(data_path, data_type.value + ".csv"))

        self.csv = self.csv[self.csv["id"].apply(lambda img_id: not self._processed_file_exists(img_id))].reset_index(drop=True)
        print(f"{len(self.csv)} files to process left.")

    def _processed_file_exists(self, img_id):
        # Define the path where 'slf_result.npy' should be saved for each image
        processed_file_path = Path(self.images_path) / img_id / "slf_result.npy"
        return processed_file_path.exists()

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        images_directory_path = os.path.join(self.images_path, self.csv["id"][index])
        return self.image_info.load_tensor(images_directory_path)[0], images_directory_path
