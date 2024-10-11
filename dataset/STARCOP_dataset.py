import glob
import os

import cv2
import pandas as pd

from torch.utils.data import Dataset
from .dataset_info import SpectralImageInfo
from .dataset_type import DatasetType


class STARCOPDataset(Dataset):
    """
    Class is used for custom dataloader. Loads images based on CSV description of data.
    """
    def __init__(self, data_path: str, data_type: DatasetType):
        self.images_path = os.path.join(data_path, data_type.get_folder_name(), data_type.get_folder_name())
        self.csv = pd.read_csv(os.path.join(data_path, data_type.value + ".csv"))

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        images_directory_path = os.path.join(self.images_path, self.csv["id"][index])

        images_AVIRIS = self.load_images(images_directory_path, "TOA_AVIRIS*.tif")
        images_WV3 = self.load_images(images_directory_path, "TOA_WV3*.tif")
        mag1c = self.load_mag1c(images_directory_path)
        labels = self.load_labels(images_directory_path, index)

        return SpectralImageInfo(images_AVIRIS, images_WV3, mag1c, labels).to_tensor()

    @staticmethod
    def load_images(path: str, pattern: str):
        return [cv2.imread(file, cv2.IMREAD_UNCHANGED) for file in glob.glob(os.path.join(path, pattern))]

    @staticmethod
    def load_mag1c(path: str):
        return {
            "weight": cv2.imread(os.path.join(path, "weight_mag1c.tif"), cv2.IMREAD_UNCHANGED),
            "mag1c": cv2.imread(os.path.join(path, "mag1c.tif"), cv2.IMREAD_UNCHANGED),
        }

    def load_labels(self, path: str, index: int):
        return {
            "label_rgba": cv2.imread(os.path.join(path, "label_rgba.tif"), cv2.IMREAD_UNCHANGED),
            "label_binary": cv2.imread(os.path.join(path, "labelbinary.tif"), cv2.IMREAD_UNCHANGED),
            #"label_string": 1.0 if self.csv["has_plume"][index] == "True" else 0.0,
        }