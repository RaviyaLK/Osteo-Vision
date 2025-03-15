import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

print(tf.__version__)
model = load_model('Train/knee_osteoporosis_model_V2.h5')

# Compile the model again (to avoid warnings)
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

def grad_cam(input_model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        inputs=input_model.inputs, outputs=[input_model.get_layer(layer_name).output, input_model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        # Ensure predictions are treated as a tensor or array
        predictions = tf.convert_to_tensor(predictions) if isinstance(predictions, list) else predictions

        print("Predictions shape:", predictions.shape)  # Debugging line
        # Changed line to access the correct index for binary classification
        loss = predictions[:, np.argmax(predictions[0])]
    grads = tape.gradient(loss, conv_outputs)

    cast_conv_outputs = tf.cast(conv_outputs > 0, "float32")
    cast_grads = tf.cast(grads > 0, "float32")
    guided_grads = cast_conv_outputs * cast_grads * grads
    weights = tf.reduce_mean(guided_grads, axis=(0, 1, 2))
    cam = np.zeros(conv_outputs[0].shape[0:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_outputs[0, :, :, i]
    cam = cv2.resize(cam.numpy(), (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / cam.max() if cam.max() > 0 else cam  # Prevent division by zero
    return heatmap


def overlay_gradcam(img, heatmap):
    # Resize heatmap to match the dimensions of the original image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # Resize to the original image dimensions
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert img to RGB if it's in BGR format
    if img.shape[2] == 3:  # Check if img has 3 channels (BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img  # If already in RGB or different format

    output = cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)
    return output

# Test Model and Display Grad-CAM
def test_model(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg19.preprocess_input(img_array)

    # Predict and Grad-CAM
    prediction = model.predict(img_array)
    print("Prediction:", prediction)

    heatmap = grad_cam(model, img_array, layer_name="block5_conv4")
    overlay_img = overlay_gradcam(cv2.imread(image_path), heatmap)

    # Display original image and heatmap side by side
    plt.figure(figsize=(10, 5))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    # Overlay Image with Heatmap
    plt.subplot(1, 2, 2)
    plt.imshow(overlay_img)
    plt.title(f"Prediction: {'Osteoporosis' if np.argmax(prediction) == 1 else 'Healthy'}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Test Example
test_model("test_images/106.jpeg")
