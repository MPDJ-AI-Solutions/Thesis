import os
import pandas as pd

from torch.utils.data import Dataset
from .dataset_info import SpectralImageInfo
from .dataset_type import DatasetType
from typing import Type


class STARCOPDataset(Dataset):
    """
    Class is used for custom dataloader. Loads images based on CSV description of data.
    """
    def __init__(self, data_path: str, data_type: DatasetType, image_info_class: Type[SpectralImageInfo]):
        self.images_path = os.path.join(data_path, data_type.get_folder_name(), data_type.get_folder_name())
        self.image_info = image_info_class
        self.csv = pd.read_csv(os.path.join(data_path, data_type.value + ".csv"))

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        images_directory_path = os.path.join(self.images_path, self.csv["id"][index])

        return self.image_info.load_tensor(images_directory_path)
