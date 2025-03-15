import numpy as np
import tensorflow as tf
from PIL import Image

def preprocess_image(image: Image.Image, model_name: str):
    if model_name == "efficientnet":
        target_size = (380, 380)
        preprocess_func = tf.keras.applications.efficientnet.preprocess_input
    else:  # VGG19 default
        target_size = (224, 224)
        preprocess_func = tf.keras.applications.vgg19.preprocess_input

    image = image.resize(target_size)
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_func(img_array)

    return img_array, target_size
