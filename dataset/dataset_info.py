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
    def load_tensor(path: str, grid_id: int = 0):
        raise NotImplementedError

class FilteredSpectralImageInfo(SpectralImageInfo):
    @staticmethod
    def load_tensor(path: str, grid_id: int = 0):
        """
        Loads tensor....

        - path: str

        """
        images_AVIRIS = [
            cv2.imread(file, cv2.IMREAD_UNCHANGED) for file in glob.glob(os.path.join(path, "TOA_AVIRIS*.tif"))
        ]

        filtered_image_path = os.path.join(path, "slf_result.npy")
        if os.path.isfile(filtered_image_path):
            filtered_image = np.load(filtered_image_path)
        else:
            filtered_image = np.zeros((512, 512), dtype=np.float32)

        mag1c = cv2.imread(os.path.join(path, "mag1c.tif"), cv2.IMREAD_UNCHANGED),


        label_rgba = cv2.imread(os.path.join(path, "label_rgba.tif"), cv2.IMREAD_UNCHANGED)
        label_binary = cv2.imread(os.path.join(path, "labelbinary.tif"), cv2.IMREAD_UNCHANGED)

        x_start = grid_id // 2 * 256
        x_end = (grid_id // 2 + 1) * 256
        y_start = (grid_id % 2) * 256
        y_end = (grid_id % 2 + 1) * 256

        tensor_AVIRIS = torch.tensor(np.array(images_AVIRIS), dtype=torch.float32)[:, x_start:x_end, y_start:y_end]
        tensor_filtered_image = torch.tensor(np.array(filtered_image), dtype=torch.float32).unsqueeze(0)[:, x_start:x_end, y_start:y_end]
        tensor_mag1c = torch.tensor(np.array(mag1c), dtype=torch.float32)[:, x_start:x_end, y_start:y_end]
        tensor_labels_rgba = torch.tensor(np.array(label_rgba), dtype=torch.float32).permute(2, 0, 1)[:,x_start:x_end, y_start:y_end]

        part_label_binary = label_binary[x_start:x_end, y_start:y_end]
        tensor_labels_binary = torch.tensor(np.array(part_label_binary), dtype=torch.float32).unsqueeze(0)
        tensor_bboxes, tensor_bboxes_labels = FilteredSpectralImageInfo.add_bbox(part_label_binary, 256, 256)

        return tensor_AVIRIS, tensor_filtered_image, tensor_mag1c, tensor_labels_rgba, tensor_labels_binary, tensor_bboxes, tensor_bboxes_labels

    @staticmethod
    def add_bbox(image, height, width, num_queries = 10):
        """
            Extract bounding boxes and labels from an image using contours.

            Args:
                image (numpy.ndarray): Binary mask image (H, W).
                height (int): Height of the image.
                width (int): Width of the image.
                num_queries (int): Maximum number of queries (default=100).

            Returns:
                result_bbox (torch.Tensor): Normalized bounding boxes (num_queries, 4).
                result_labels (torch.Tensor): Labels for each query (num_queries).
        """
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result_bbox = torch.zeros((num_queries, 4), dtype=torch.float32)
        result_labels = torch.zeros((num_queries,), dtype=torch.int64)

        for i, contour in enumerate(contours[:num_queries]):
            x, y, w, h = cv2.boundingRect(contour)

            cx = (x + w / 2) / width
            cy = (y + h / 2) / height
            w_scaled = w / width
            h_scaled = h / height

            result_bbox[i, :] = torch.tensor([cx, cy, w_scaled, h_scaled], dtype=torch.float32)
            result_labels[i] = 1


        return result_bbox, result_labels