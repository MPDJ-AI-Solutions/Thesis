import torch
from sklearn.cluster import KMeans
from torch import nn


class SpectralLinearFilter(nn.Module):
    """
    TODO: COMMENT
    """
    def __init__(self, num_classes: int = 20, min_class_size: int = 10000):
        super(SpectralLinearFilter, self).__init__()
        self.num_classes = num_classes
        self.min_class_size = min_class_size


    @staticmethod
    def compute_segmentation(image: torch.Tensor, num_classes) -> torch.Tensor:
        pixels = image.contiguous().view(-1, image.size(2))

        kmeans = KMeans(n_clusters=num_classes, random_state=0)
        segmentation_mask = kmeans.fit_predict(pixels).reshape(image.shape[0], image.shape[1])

        return torch.from_numpy(segmentation_mask)


    @staticmethod
    def compute_covariance(pixels: torch.Tensor, mean_vector) -> torch.Tensor:
        num_pixels = len(pixels)
        centered_pixels = pixels - mean_vector
        covariance = (centered_pixels.T @ centered_pixels) / num_pixels

        return covariance


    @staticmethod
    def spectral_linear_filter(pixel_spectrum, mean_vector, inv_covariance_matrix, methane_pattern) -> torch.Tensor:
        centered_spectrum = pixel_spectrum - mean_vector
        numerator = (centered_spectrum.T @ inv_covariance_matrix @ methane_pattern)
        denominator = torch.sqrt(methane_pattern.T @ inv_covariance_matrix @ methane_pattern)

        return numerator / denominator


    def forward(self, hyperspectral_image, methane_pattern) -> torch.Tensor:
        batch_size, num_channels, height, width = hyperspectral_image.shape

        result = torch.zeros((batch_size, height, width), dtype=torch.float32)
        for batch in range(batch_size):
            image = hyperspectral_image[batch].permute(1, 2, 0)
            segmentation_mask = self.compute_segmentation(image=image, num_classes=self.num_classes)

            methane_concentration_map = torch.zeros((height, width))

            for class_index in range(self.num_classes):
                class_pixel_indices = (segmentation_mask == class_index).nonzero(as_tuple=True)
                class_pixel = image[class_pixel_indices]

                if class_pixel.shape[0] < self.min_class_size:
                    continue

                mean_vector = class_pixel.mean(dim=0)
                covariance_matrix = self.compute_covariance(pixels=class_pixel, mean_vector=mean_vector)
                inv_covariance_matrix = torch.inverse(covariance_matrix)

                for idx in zip(*class_pixel_indices):
                    pixel_spectrum = image[idx]
                    methane_concentration = self.spectral_linear_filter(
                        pixel_spectrum, mean_vector, inv_covariance_matrix, methane_pattern
                    )
                    methane_concentration_map[idx] = methane_concentration

            result[batch] = methane_concentration_map

        return result
