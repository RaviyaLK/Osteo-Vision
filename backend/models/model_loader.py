import tensorflow as tf

MODEL_PATHS = {
    "binary_vgg19": "Train/knee_osteoporosis_model_V2_new_new.h5",
    "efficientnet": "Train/efficientnet_binary_model.h5",
    "multiclass_vgg19": "Train/multiclass_vgg19_model.h5"
}

def load_model(model_name):
    path = MODEL_PATHS.get(model_name)
    if path is None:
        raise ValueError(f"Invalid model name: {model_name}")

    model = tf.keras.models.load_model(path, compile=False)

    # Choose loss function based on binary/multiclass
    if model_name == "multiclass_vgg19":
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
