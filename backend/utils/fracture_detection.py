import cv2
import numpy as np
from PIL import Image
import io

def detect_fracture(image: np.ndarray):
    """
    Detects fractures in an X-ray image using Canny Edge Detection.
    Returns the processed image with edges highlighted.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply Canny Edge Detection (adjust thresholds if needed)
    edges = cv2.Canny(blurred, 50, 150)

    return edges
