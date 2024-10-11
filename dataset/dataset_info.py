import torch
import numpy as np


class SpectralImageInfo:
    """
    Class used to gather associated information about a data record.
    """
    def __init__(self, images_AVIRIS, images_WV3, mag1c, labels):
        self.images_AVIRIS = images_AVIRIS
        self.images_WV3 = images_WV3
        self.mag1c = mag1c
        self.labels = labels

    def to_tensor(self):
        tensor_AVIRIS = torch.tensor(np.array(self.images_AVIRIS), dtype=torch.float32)
        tensor_WV3 = torch.tensor(np.array(self.images_WV3), dtype=torch.float32)

        tensor_mag1c = torch.tensor(np.array(list(self.mag1c.values())), dtype=torch.float32)
        tensor_labels_rgba = torch.tensor(np.array(list(self.labels["label_rgba"])), dtype=torch.float32).permute(2, 0, 1)
        tensor_labels_binary = torch.tensor(np.array(list(self.labels["label_binary"])), dtype=torch.float32).unsqueeze(0)

        return torch.cat((tensor_AVIRIS, tensor_WV3, tensor_mag1c, tensor_labels_rgba, tensor_labels_binary))


def transformer_input_converter(tensor):
    return tensor[:, :8, :, :], tensor[:, 16:20, :, :]
