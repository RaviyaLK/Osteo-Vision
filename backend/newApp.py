from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from skimage.feature import graycomatrix, graycoprops
import numpy as np
import tensorflow as tf
import cv2
import uvicorn
from PIL import Image
import io
import base64

app = FastAPI()

# CORS Configuration for frontend
origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
MODEL_PATH = "Train/knee_osteoporosis_model_V2.h5"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Grad-CAM Function
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

# Overlay Grad-CAM on Image
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


def estimate_bone_density(image: np.ndarray):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    mean_intensity = np.mean(gray_image)  # Calculate mean pixel intensity

    # Define thresholds for estimation (adjust based on dataset)
    if mean_intensity > 180:
        density_category = "High Bone Density"
    elif mean_intensity > 120:
        density_category = "Normal Bone Density"
    else:
        density_category = "Low Bone Density (Possible Osteoporosis)"

    return mean_intensity, density_category

@app.post("/bone_density/")
async def bone_density_analysis(file: UploadFile = File(...)):
    try:
        # Read file as bytes
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))  # Resize to match model input size
        img_array = np.array(image)

        # Estimate bone density
        mean_intensity, density_category = estimate_bone_density(img_array)

        return JSONResponse(content={
            "mean_pixel_intensity": mean_intensity,
            "density_category": density_category
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
def measure_cortical_thickness(image: np.ndarray):
    """
    Measures cortical bone thickness using edge detection.
    Returns average thickness and estimated osteoporosis risk.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred, 100, 200)

    # Find contours of the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return {"error": "No bone edges detected. Image might be unclear."}

    # Find the two main bone edges (assumed cortical layers)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get bounding box around the bone
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Sample multiple points along the bone
    thicknesses = []
    for i in range(y, y + h, 10):  # Sample every 10 pixels vertically
        edge_pixels = np.where(edges[i, x:x + w] > 0)[0]  # Find edges in row
        if len(edge_pixels) >= 2:  # Ensure two edges are found
            thickness = edge_pixels[-1] - edge_pixels[0]  # Difference between edges
            thicknesses.append(thickness)

    if not thicknesses:
        return {"error": "Failed to measure cortical thickness."}

    # Compute average cortical thickness
    avg_thickness = np.mean(thicknesses)

    # Osteoporosis risk classification (estimated thresholds)
    if avg_thickness > 3.5:  # Thick cortical bone
        risk = "Low"
    elif 2.0 <= avg_thickness <= 3.5:  # Moderate thickness
        risk = "Moderate"
    else:  # Very thin cortical bone
        risk = "High"

    return {
        "average_cortical_thickness": round(avg_thickness, 2),
        "osteoporosis_risk": risk
    }
@app.post("/cortical_thickness/")
async def cortical_thickness_api(file: UploadFile = File(...)):
    """
    API to process an X-ray image and measure cortical bone thickness.
    """
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = np.array(image)

        # Measure cortical thickness
        result = measure_cortical_thickness(image)

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Read file as bytes
        image_bytes = await file.read()
        if not image_bytes:
            return JSONResponse(status_code=400, content={"error": "Empty file uploaded"})

        # Open image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))
        img_array = np.array(image)

        # Preprocessing (Matching standalone script)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.vgg19.preprocess_input(img_array)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = "Osteoporosis" if np.argmax(prediction) == 1 else "Healthy"

        # Generate Grad-CAM
        heatmap = grad_cam(model, img_array)
        overlay_img = overlay_gradcam(img_bgr, heatmap)

        # Convert overlay image to base64 for response
        _, buffer = cv2.imencode(".jpg", overlay_img)
        overlay_base64 = base64.b64encode(buffer).decode()

        return JSONResponse(content={"prediction": predicted_class, "gradcam_image": overlay_base64})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
