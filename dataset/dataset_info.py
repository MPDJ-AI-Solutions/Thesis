import glob
import os

import cv2
import torch
import numpy as np


class SpectralImageInfo:
    """
    Class used to gather associated information about a data record.
    """
    @staticmethod
    def load_tensor(path: str):
        raise NotImplementedError


class TransformerModelSpectralImageInfo(SpectralImageInfo):
    @staticmethod
    def load_tensor(path: str):
        images_AVIRIS = [
            cv2.imread(file, cv2.IMREAD_UNCHANGED) for file in glob.glob(os.path.join(path, "TOA_AVIRIS*.tif"))
        ]

        mag1c = [
            cv2.imread(os.path.join(path, "mag1c.tif"), cv2.IMREAD_UNCHANGED),
            cv2.imread(os.path.join(path, "weight_mag1c.tif"), cv2.IMREAD_UNCHANGED),
        ]

        label_rgba = cv2.imread(os.path.join(path, "label_rgba.tif"), cv2.IMREAD_UNCHANGED)
        label_binary = cv2.imread(os.path.join(path, "labelbinary.tif"), cv2.IMREAD_UNCHANGED)

        tensor_AVIRIS = torch.tensor(np.array(images_AVIRIS), dtype=torch.float32)
        tensor_mag1c = torch.tensor(np.array(mag1c), dtype=torch.float32)
        tensor_labels_rgba = torch.tensor(np.array(label_rgba), dtype=torch.float32).permute(2, 0, 1)
        tensor_labels_binary = torch.tensor(np.array(label_binary), dtype=torch.float32).unsqueeze(0)

        return torch.cat((tensor_AVIRIS, tensor_mag1c, tensor_labels_rgba, tensor_labels_binary))

    @staticmethod
    def backbone_input_converter(tensor):
        # Currently without mag1c
        return tensor[:, :8, :, :]
