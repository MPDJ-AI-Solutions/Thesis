import torch
import torch.multiprocessing as mp

from sklearn.cluster import KMeans
from torch import nn


class SpectralLinearFilter(nn.Module):
    """
    A PyTorch module for spectral linear filtering of hyperspectral images to detect methane concentration.
    """

    def __init__(self, num_classes: int = 20, min_class_size: int = 10000):
        """
        Initializes the SpectralLinearFilter with the specified number of classes and minimum class size.

        Args:
            num_classes (int): The number of classes to generate. Default is 20.
            min_class_size (int): The minimum size of each class. Default is 10000.
        """
        super(SpectralLinearFilter, self).__init__()
        self.num_classes = num_classes
        self.min_class_size = min_class_size


    @staticmethod
    def compute_segmentation(image: torch.Tensor, num_classes) -> torch.Tensor:
        """
        Computes the segmentation mask for a given image using K-means clustering.

        Args:
            image (torch.Tensor): The input image tensor of shape (H, W, C), where H is the height,
                                  W is the width, and C is the number of channels.
            num_classes (int): The number of clusters/classes for segmentation.
        Returns:
            torch.Tensor: A tensor containing the segmentation mask of shape (H, W), where each
                          pixel value represents the cluster/class it belongs to.
        """
        pixels = image.to("cpu").contiguous().view(-1, image.size(2))
        kmeans = KMeans(n_clusters=num_classes, random_state=0)
        segmentation_mask = kmeans.fit_predict(pixels).reshape(image.shape[0], image.shape[1])
        return torch.from_numpy(segmentation_mask)


    @staticmethod
    def compute_covariance(pixels: torch.Tensor, mean_vector) -> torch.Tensor:
        """
        Compute the covariance matrix of the given pixel data.

        Args:
            pixels (torch.Tensor): A tensor containing pixel data with shape (num_pixels, num_features).
            mean_vector (torch.Tensor): A tensor containing the mean vector with shape (num_features,).
        Returns:
            torch.Tensor: The covariance matrix with shape (num_features, num_features).
        """
        num_pixels = len(pixels)
        centered_pixels = pixels - mean_vector
        covariance = (centered_pixels.T @ centered_pixels) / num_pixels
        regularization_term = torch.eye(covariance.size(0)) * 1e-6  # Small value on diagonal
        covariance += regularization_term  # Add to make matrix invertible
        return covariance


    @staticmethod
    def spectral_linear_filter(pixel_spectrum, mean_vector, inv_covariance_matrix, methane_pattern) -> torch.Tensor:
        """
        Applies a spectral linear filter to the given pixel spectrum.

        Args:
            pixel_spectrum (torch.Tensor): The spectrum of the pixel to be filtered.
            mean_vector (torch.Tensor): The mean vector of the spectra.
            inv_covariance_matrix (torch.Tensor): The inverse covariance matrix of the spectra.
            methane_pattern (torch.Tensor): The pattern vector for methane detection.
        Returns:
            torch.Tensor: The result of the spectral linear filter applied to the pixel spectrum.
        """
        centered_spectrum = pixel_spectrum - mean_vector
        numerator = (centered_spectrum.T @ inv_covariance_matrix @ methane_pattern)
        denominator = torch.sqrt(methane_pattern.T @ inv_covariance_matrix @ methane_pattern)
        return numerator / denominator


    def forward(self, hyperspectral_image, methane_pattern) -> torch.Tensor:
        """
        Forward pass for the spectral linear filter.

        Args:
            hyperspectral_image (torch.Tensor): A 4D tensor of shape (batch_size, num_channels, height, width)
                representing the hyperspectral images.
            methane_pattern (numpy.ndarray or list): A 1D array or list representing the methane pattern.
        Returns:
            torch.Tensor: A 3D tensor of shape (batch_size, height, width) representing the methane concentration maps.
        """
        batch_size, num_channels, height, width = hyperspectral_image.shape
        hyperspectral_image = hyperspectral_image.contiguous()
        methane_pattern = torch.FloatTensor(methane_pattern)
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



class SpectralLinearFilterParallel(SpectralLinearFilter):
    """
    Parallelized version of the SpectralLinearFilter class
    """
    def __init__(self, num_classes: int = 20, min_class_size: int = 10000):
        super(SpectralLinearFilterParallel, self).__init__(num_classes, min_class_size)

    @staticmethod
    def process_image(batch, hyperspectral_image, methane_pattern, num_classes, min_class_size, height, width):
        image = hyperspectral_image[batch].permute(1, 2, 0)
        segmentation_mask = SpectralLinearFilter.compute_segmentation(image=image, num_classes=num_classes)
        methane_concentration_map = torch.zeros((height, width), dtype=torch.float32)

        for class_index in range(num_classes):
            class_pixel_indices = (segmentation_mask == class_index).nonzero(as_tuple=True)
            class_pixel = image[class_pixel_indices]

            if class_pixel.shape[0] < min_class_size:
                continue

            mean_vector = class_pixel.mean(dim=0)
            covariance_matrix = SpectralLinearFilter.compute_covariance(pixels=class_pixel, mean_vector=mean_vector)
            inv_covariance_matrix = torch.inverse(covariance_matrix)

            for idx in zip(*class_pixel_indices):
                pixel_spectrum = image[idx]
                methane_concentration = SpectralLinearFilter.spectral_linear_filter(
                    pixel_spectrum, mean_vector, inv_covariance_matrix, methane_pattern
                )
                methane_concentration_map[idx] = methane_concentration

        return batch, methane_concentration_map

    def forward(self, hyperspectral_image, methane_pattern) -> torch.Tensor:
        """
        Forward pass for the spectral linear filter.
        
        Args:
            hyperspectral_image (torch.Tensor): A 4D tensor of shape (batch_size, num_channels, height, width)
                representing the hyperspectral images.
            methane_pattern (numpy.ndarray or list): A 1D array or list representing the methane pattern.
        Returns:
            torch.Tensor: A 3D tensor of shape (batch_size, height, width) representing the methane concentration maps.
        """
        batch_size, num_channels, height, width = hyperspectral_image.shape
        hyperspectral_image = hyperspectral_image.contiguous()
        methane_pattern = torch.FloatTensor(methane_pattern)

        # Shared memory tensor to hold results
        result = torch.zeros((batch_size, height, width), dtype=torch.float32)

        # Create a multiprocessing pool to parallelize batch processing
        with mp.Pool() as pool:
            # Map each image in the batch to be processed in parallel
            result_list = pool.starmap(
                self.process_image,
                [(batch, hyperspectral_image, methane_pattern, self.num_classes, self.min_class_size, height, width) for batch in range(batch_size)]
            )

            # Store the results back into the result tensor
            for batch, methane_concentration_map in result_list:
                result[batch] = methane_concentration_map

        return result
