import tensorflow as tf

MODEL_PATHS = {
    "binary_vgg19": "model_files/knee_osteoporosis_model_V2.h5",
    "efficientnet": "model_files/knee_osteoporosis_efficientnet_model.h5",
    "multiclass_vgg19": "model_files/knee_osteoporosis_multiclass_model.h5"
}

def load_model(model_name):
    path = MODEL_PATHS.get(model_name)
    if path is None:
        raise ValueError(f"Invalid model name: {model_name}")
    model = tf.keras.models.load_model(path, compile=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
