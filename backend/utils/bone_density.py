import numpy as np
import cv2

def estimate_bone_density(image):
    """Estimates bone density using grayscale intensity analysis."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean_intensity = np.mean(gray_image)

    if mean_intensity > 180:
        density_category = "High Bone Density"
    elif mean_intensity > 120:
        density_category = "Normal Bone Density"
    else:
        density_category = "Low Bone Density (Possible Osteoporosis)"

    return mean_intensity, density_category
