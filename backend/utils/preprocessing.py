import numpy as np
import tensorflow as tf
from PIL import Image

def preprocess_image(image: Image.Image, model_name: str):
    if model_name == "efficientnet":
        target_size = (224, 224)
        preprocess_func = tf.keras.applications.efficientnet.preprocess_input
    else:  # VGG19 models
        target_size = (224, 224)
        preprocess_func = tf.keras.applications.vgg19.preprocess_input

    # Resize the image to the required model input size
    image_resized = image.resize(target_size)

    # Convert to NumPy array
    img_array = np.array(image_resized)

    # Expand dimensions to match model input shape
    img_array = np.expand_dims(img_array, axis=0)

    # Apply model-specific preprocessing
    img_array = preprocess_func(img_array)

    return img_array, target_size, image_resized
