import glob
import os

import cv2
import pandas as pd

from enum import Enum
from dataset_info import DatasetInfo


class DatasetType(Enum):
    TRAIN = "train"
    TEST = "test"
    UNITTEST = "ut"

    def describe(self):
        return f"Dataset is in state: {self.name}"
    
    def get_folder_name(self):
        return f"STARCOP_{self.name}"


class STARCOPDataset:
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

        images_AVIRIS = [cv2.imread(file, cv2.IMREAD_UNCHANGED) for file
                         in glob.glob(os.path.join(images_directory_path, "TOA_AVIRIS*.tif"))]
        images_WV3 = [cv2.imread(file, cv2.IMREAD_UNCHANGED) for file
                      in glob.glob(os.path.join(images_directory_path, "TOA_WV3*.tif"))]

        mag1c = {
            "weight": cv2.imread(os.path.join(images_directory_path, "weight_mag1c.tif"), cv2.IMREAD_UNCHANGED),
            "mag1c": cv2.imread(os.path.join(images_directory_path, "mag1c.tif"), cv2.IMREAD_UNCHANGED),
        }

        labels = {
            "label_rgba": cv2.imread(os.path.join(images_directory_path, "label_rgba.tif"), cv2.IMREAD_UNCHANGED),
            "label_binary": cv2.imread(os.path.join(images_directory_path, "labelbinary.tif"), cv2.IMREAD_UNCHANGED),
            "label_string": self.csv["has_plume"][index]
        }

        return DatasetInfo(images_AVIRIS, images_WV3, mag1c, labels)

