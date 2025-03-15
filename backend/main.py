from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from utils.fracture_detection import detect_fracture
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import numpy as np
import cv2
import base64
from PIL import Image
import io
import tensorflow as tf
from models.model_loader import load_model
from utils.gradcam import overlay_gradcam, grad_cam
from utils.preprocessing import preprocess_image
from utils.bone_density import estimate_bone_density
from utils.cortical_thickness import measure_cortical_thickness

# Initialize FastAPI app
app = FastAPI()

# CORS Configuration
origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = load_model()

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...), model: str = "binary_vgg19"):
    try:
        image_bytes = await file.read()
        if not image_bytes:
            return JSONResponse(status_code=400, content={"error": "Empty file uploaded"})

        # Load selected model dynamically
        model_instance = load_model(model)

        # Open and preprocess image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array, target_size = preprocess_image(image, model)
        img_for_overlay = np.array(image.resize(target_size))

        if model == "efficientnet":
            img_bgr = cv2.cvtColor(img_for_overlay, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = cv2.cvtColor(img_for_overlay, cv2.COLOR_RGB2BGR)

        # Predict
        prediction = model_instance.predict(img_array)[0]
        predicted_class_idx = int(np.argmax(prediction)) if model == "multiclass_vgg19" else int(round(prediction[0]))
        confidence_score = round(float(np.max(prediction)), 2)

        if model == "multiclass_vgg19":
            class_names = ["Healthy", "Osteopenia", "Osteoporosis"]
            predicted_class = class_names[predicted_class_idx]
        else:
            predicted_class = "Osteoporosis" if predicted_class_idx == 1 else "Healthy"

        # Grad-CAM
        heatmap = grad_cam(model_instance, img_array)
        overlay_img = overlay_gradcam(img_bgr, heatmap)

        # Encode image
        _, buffer = cv2.imencode(".jpg", overlay_img)
        overlay_base64 = base64.b64encode(buffer).decode()

        return JSONResponse(content={
            "prediction": predicted_class,
            "confidence": confidence_score,
            "gradcam_image": overlay_base64
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
@app.post("/bone_density/")
async def bone_density_analysis(file: UploadFile = File(...)):
    """Analyzes bone density from an X-ray image."""
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(image)

        mean_intensity, density_category = estimate_bone_density(img_array)
        return JSONResponse(content={"mean_pixel_intensity": mean_intensity, "density_category": density_category})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/cortical_thickness/")
async def cortical_thickness_api(file: UploadFile = File(...)):
    """Processes an X-ray image and measures cortical bone thickness."""
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(image)

        result = measure_cortical_thickness(img_array)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
@app.post("/detect_fracture/")
async def fracture_detection_api(file: UploadFile = File(...)):
    """
    API to process an X-ray image and detect fractures using edge detection.
    """
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = np.array(image)

        # Detect fractures
        edges = detect_fracture(image)

        # Convert edges to PNG format for response
        _, encoded_image = cv2.imencode(".png", edges)
        return Response(content=encoded_image.tobytes(), media_type="image/png")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
