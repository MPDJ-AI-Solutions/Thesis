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


class TransformerModelSpectralImageInfo(SpectralImageInfo):
    @staticmethod
    def load_tensor(path: str, image: int = 0):
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
        tensor_bboxes = torch.tensor(TransformerModelSpectralImageInfo.add_bbox(label_binary), dtype=torch.float32).permute(2, 0, 1)

        return torch.cat((tensor_AVIRIS, tensor_mag1c, tensor_labels_rgba, tensor_labels_binary, tensor_bboxes), dim=0)

    @staticmethod
    def backbone_input_converter(tensor):
        # Currently without mag1c
        return tensor[:, :8, :, :], tensor[:, 14, :, :]

    @staticmethod
    def get_bbox(tensor):
        return tensor[:, 15:18, :, :].permute(0, 2, 3, 1)

    @staticmethod
    def add_bbox(image):
        bbox_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(bbox_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return bbox_image


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
        y_start = (grid_id % 4) * 128,
        y_end = (grid_id % 4 + 1) * 128

        tensor_AVIRIS = torch.tensor(np.array(images_AVIRIS), dtype=torch.float32)[x_start:x_end, y_start:y_end, :]
        tensor_filtered_image = torch.tensor(np.array(filtered_image), dtype=torch.float32).unsqueeze(0)[x_start:x_end, y_start:y_end, :]
        tensor_mag1c = torch.tensor(np.array(mag1c), dtype=torch.float32)[x_start:x_end, y_start:y_end, :]
        tensor_labels_rgba = torch.tensor(np.array(label_rgba), dtype=torch.float32).permute(2, 0, 1)[x_start:x_end, y_start:y_end, :]
        tensor_labels_binary = torch.tensor(np.array(label_binary), dtype=torch.float32).unsqueeze(0)[x_start:x_end, y_start:y_end, :]
        tensor_bboxes, tensor_bboxes_confidence = FilteredSpectralImageInfo.add_bbox(tensor_labels_binary, 512, 512)

        return tensor_AVIRIS, tensor_filtered_image, tensor_mag1c, tensor_labels_rgba, tensor_labels_binary, tensor_bboxes, tensor_bboxes_confidence

    @staticmethod
    def add_bbox(image, height, width):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result_bbox = torch.zeros((16, 4))
        result_confidence = torch.zeros((16, 2))
        i = 0
        for contour in contours[:16]:
            x, y, w, h = cv2.boundingRect(contour)

            x_scaled = x / 512
            y_scaled = y / 512
            w_scaled = w / 512
            h_scaled = h / 512

            result_bbox[i, :] = torch.tensor([x_scaled, y_scaled, w_scaled, h_scaled], dtype=torch.float32)
            result_confidence[i] = torch.tensor([0, 1], dtype=torch.float32)
            i += 1

        for j in range(16 - i):
            result_confidence[i + j] = torch.tensor([1, 0], dtype=torch.float32)


        return result_bbox, result_confidence