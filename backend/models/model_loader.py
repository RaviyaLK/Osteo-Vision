import tensorflow as tf






MODEL_PATHS = {
    "binary_vgg19": "Train/knee_osteoporosis_model_V6.h5",
    "efficientnet": "Train/knee_osteoporosis_model_efficientNet.h5",
    "multiclass_vgg19": "Train/Multiclass_Osteoporosis_Model_new.h5"
}

def load_model(model_name):
    path = MODEL_PATHS.get(model_name)
    if path is None:
        raise ValueError(f"Invalid model name: {model_name}")
    model = tf.keras.models.load_model(path, compile=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
