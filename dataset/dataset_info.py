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
            filtered_image = np.zeros((128, 128, 1), dtype=np.float32)

        mag1c = cv2.imread(os.path.join(path, "mag1c.tif"), cv2.IMREAD_UNCHANGED),


        label_rgba = cv2.imread(os.path.join(path, "label_rgba.tif"), cv2.IMREAD_UNCHANGED)
        label_binary = cv2.imread(os.path.join(path, "labelbinary.tif"), cv2.IMREAD_UNCHANGED)

        x_start = grid_id // 4 * 128
        x_end = (grid_id // 4 + 1) * 128
        y_start = (grid_id % 4) * 128
        y_end = (grid_id % 4 + 1) * 128



        tensor_AVIRIS = torch.tensor(np.array(images_AVIRIS), dtype=torch.float32)[:, x_start:x_end, y_start:y_end]
        tensor_filtered_image = torch.tensor(np.array(filtered_image), dtype=torch.float32).unsqueeze(0)[:, x_start:x_end, y_start:y_end]
        tensor_mag1c = torch.tensor(np.array(mag1c), dtype=torch.float32)[:, x_start:x_end, y_start:y_end]
        tensor_labels_rgba = torch.tensor(np.array(label_rgba), dtype=torch.float32).permute(2, 0, 1)[:,x_start:x_end, y_start:y_end]

        array = label_binary[x_start:x_end, y_start:y_end]
        tensor_labels_binary = torch.tensor(np.array(label_binary), dtype=torch.float32).unsqueeze(0)
        tensor_bboxes, tensor_bboxes_confidence = FilteredSpectralImageInfo.add_bbox(array, 128, 128)

        return tensor_AVIRIS, tensor_filtered_image, tensor_mag1c, tensor_labels_rgba, tensor_labels_binary, tensor_bboxes, tensor_bboxes_confidence

    @staticmethod
    def add_bbox(image, height, width, num_queries = 1):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result_bbox = torch.zeros((num_queries, 4))
        result_confidence = torch.zeros((num_queries, 1))
        i = 0
        for contour in contours[:num_queries]:
            x, y, w, h = cv2.boundingRect(contour)

            x_scaled = x / width
            y_scaled = y / height
            w_scaled = (x + w) / width
            h_scaled = (y + h) / height

            result_bbox[i, :] = torch.tensor([x_scaled, y_scaled, w_scaled, h_scaled], dtype=torch.float32)
            result_confidence[i] = torch.tensor([1], dtype=torch.float32)
            i += 1

        for j in range(num_queries - i):
            result_confidence[i + j] = torch.tensor([0], dtype=torch.float32)


        return result_bbox, result_confidence