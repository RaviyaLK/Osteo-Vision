import cv2
import io
import base64
import numpy as np
import tensorflow as tf
from fastapi import UploadFile
from PIL import Image

def grad_cam(input_model, img_array, layer_name="block5_conv4"):
    grad_model = tf.keras.models.Model(
        inputs=input_model.inputs,
        outputs=[input_model.get_layer(layer_name).output, input_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)

    # Normalize gradients
    cast_conv_outputs = tf.cast(conv_outputs > 0, "float32")
    cast_grads = tf.cast(grads > 0, "float32")
    guided_grads = cast_conv_outputs * cast_grads * grads
    weights = tf.reduce_mean(guided_grads, axis=(0, 1, 2))

    cam = np.zeros(conv_outputs.shape[1:3], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_outputs[0, :, :, i]

    cam = cv2.resize(cam.numpy(), (224, 224))
    heatmap = np.maximum(cam, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1  # Avoid division by zero

    return heatmap

def overlay_gradcam(img, heatmap):
    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)

    # Apply threshold to extract high-activation regions
    _, threshold = cv2.threshold(heatmap_resized, 150, 255, cv2.THRESH_BINARY)

    # Find contours of high-activation areas
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ensure image is in BGR format before drawing
    if len(img.shape) == 2 or img.shape[-1] == 1:  # Grayscale image
        img_with_boxes = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_with_boxes = img.copy()  # Already in BGR/RGB format

    # Apply heatmap overlay
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    output = cv2.addWeighted(img_with_boxes, 0.6, heatmap_color, 0.4, 0)

    # Find the largest activation region
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Draw a bounding box around the highest activation
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add text annotation
        text = "Most Influential Area"
        text_position = (x, y - 10) if y - 10 > 10 else (x, y + 20)
        cv2.putText(output, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw an arrow pointing to the region
        arrow_start = (x + w // 2, y + h + 10)
        arrow_end = (x + w // 2, y + h + 30)
        cv2.arrowedLine(output, arrow_start, arrow_end, (0, 255, 0), 2)

    return output
