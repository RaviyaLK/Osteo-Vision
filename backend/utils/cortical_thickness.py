import numpy as np
import cv2

def measure_cortical_thickness(image):
    """Measures cortical bone thickness using edge detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"error": "No bone edges detected. Image might be unclear."}

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    thicknesses = []
    for i in range(y, y + h, 10):
        edge_pixels = np.where(edges[i, x:x + w] > 0)[0]
        if len(edge_pixels) >= 2:
            thickness = edge_pixels[-1] - edge_pixels[0]
            thicknesses.append(thickness)

    if not thicknesses:
        return {"error": "Failed to measure cortical thickness."}

    avg_thickness = np.mean(thicknesses)

    if avg_thickness > 3.5:
        risk = "Low"
    elif 2.0 <= avg_thickness <= 3.5:
        risk = "Moderate"
    else:
        risk = "High"

    return {"average_cortical_thickness": round(avg_thickness, 2), "osteoporosis_risk": risk}
