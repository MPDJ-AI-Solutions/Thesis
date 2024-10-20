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
        tensor_bboxes = torch.tensor(add_bbox(label_binary), dtype=torch.float32).permute(2, 0, 1)

        return torch.cat((tensor_AVIRIS, tensor_mag1c, tensor_labels_rgba, tensor_labels_binary, tensor_bboxes), dim=0)

    @staticmethod
    def backbone_input_converter(tensor):
        # Currently without mag1c
        return tensor[:, :8, :, :], tensor[:, 14, :, :]

    @staticmethod
    def get_bbox(tensor):
        return tensor[:, 15:18, :, :].permute(0, 2, 3, 1)

def add_bbox(image):
    bbox_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(bbox_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return bbox_image


