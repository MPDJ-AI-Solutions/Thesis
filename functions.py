import matplotlib.pyplot as plt
import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    image_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    plt.imshow(image_array, cmap='gray')
    plt.show()
    plt.close()
    return image_array
