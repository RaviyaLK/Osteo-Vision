import numpy as np
import cv2
import base64
import tensorflow as tf

def generate_saliency_map_image(model, img_array, image_resized):
    # Ensure we track gradients
    img_tensor = tf.convert_to_tensor(img_array)
    img_tensor = tf.cast(img_tensor, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        class_index = tf.argmax(predictions[0])
        class_score = predictions[:, class_index]

    # Compute gradient of the output class w.r.t input image
    gradients = tape.gradient(class_score, img_tensor)
    saliency = tf.reduce_max(tf.abs(gradients), axis=-1)[0]

    # Normalize the saliency map
    saliency = saliency.numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    saliency = np.uint8(saliency * 255)

    # Convert to heatmap
    saliency_colored = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
    img_rgb = cv2.cvtColor(np.array(image_resized), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_rgb, 0.6, saliency_colored, 0.4, 0)

    _, buffer = cv2.imencode(".jpg", overlay)
    return base64.b64encode(buffer).decode()
