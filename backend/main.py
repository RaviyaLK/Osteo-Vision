import base64
import io
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from config import MONGO_URL
from models.model_loader import load_model
from utils.gradcam import grad_cam, overlay_gradcam
from utils.preprocessing import preprocess_image
from inference_sdk import InferenceHTTPClient
from config import ROBOFLOW_API_URL, ROBOFLOW_API_KEY
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

# Roboflow API client
CLIENT = InferenceHTTPClient(
    api_url=ROBOFLOW_API_URL,
    api_key=ROBOFLOW_API_KEY
)

# MongoDB configuration
client = MongoClient(MONGO_URL)
db = client.reportsDB
reports_collection = db.reports

# Report data model for saving into MongoDB
class Report(BaseModel):
    patient_id: str
    patient_name: str
    model_used: str
    prediction: str
    confidence_score: float
    report_pdf: bytes  # Store the PDF as binary data

# Endpoint to save a report
from base64 import b64decode
from fastapi import HTTPException

@app.post("/save-report/")
async def save_report(
    patient_id: str = Form(...),
    patient_name: str = Form(...),
    model_used: str = Form(...),
    prediction: str = Form(...),
    confidence_score: float = Form(...),
    report_pdf: UploadFile = File(...),
):
    try:
        pdf_bytes = await report_pdf.read()  # Read the uploaded PDF as bytes

        # Prepare report data for MongoDB
        report_data = {
            "patient_id": patient_id,
            "patient_name": patient_name,
            "model_used": model_used,
            "prediction": prediction,
            "confidence_score": confidence_score,
            "report_pdf": pdf_bytes,  # Store the binary PDF data
            "generated_on": datetime.now(),
        }

        result = reports_collection.insert_one(report_data)
        return JSONResponse(content={"message": "Report saved successfully!", "id": str(result.inserted_id)})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving report: {str(e)}")

# Endpoint to get all saved reports (history)
@app.get("/get-reports/")
async def get_reports():
    reports = list(reports_collection.find({}, {"report_pdf": 0}))  # Exclude PDF binary data from response
    for report in reports:
        report['_id'] = str(report['_id'])  # Convert ObjectId to string
    return reports

# Endpoint to download a report
@app.get("/download-report/{report_id}")
async def download_report(report_id: str):
    report = reports_collection.find_one({"_id": ObjectId(report_id)})

    if not report:
        return JSONResponse(content={"message": "Report not found"}, status_code=404)

    pdf_bytes = report.get("report_pdf")
    if not pdf_bytes:
        return JSONResponse(content={"message": "No PDF available for this report"}, status_code=404)

    return StreamingResponse(io.BytesIO(pdf_bytes), media_type="application/pdf", headers={
        "Content-Disposition": f"attachment; filename={report['patient_name']}_report.pdf"
    })

# Upload image and perform model inference
@app.post("/upload/")
async def upload_image(file: UploadFile = File(...), model: str = Form("binary_vgg19")):
    try:
        image_bytes = await file.read()
        if not image_bytes:
            return JSONResponse(status_code=400, content={"error": "Empty file uploaded"})

        # Check if the image is a knee X-ray
        is_knee = await is_knee_xray(image_bytes)
        if not is_knee:
            return JSONResponse(status_code=400, content={"error": "Uploaded image is not a knee X-ray"})

        # Load selected model dynamically
        model_instance = load_model(model)

        # Open and preprocess image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array, target_size, image_resized = preprocess_image(image, model)

        img_for_overlay = np.array(image_resized)
        img_bgr = cv2.cvtColor(img_for_overlay, cv2.COLOR_RGB2BGR)

        # Predict
        prediction = model_instance.predict(img_array)[0]
        predicted_class_idx = int(np.argmax(prediction)) if model == "multiclass_vgg19" else int(round(prediction[0]))
        max_value = 0.99
        confidence_score = min(round(float(np.max(prediction)), 2), max_value)

        if model == "multiclass_vgg19":
            class_names = ["Healthy", "Osteopenia", "Osteoporosis"]
            predicted_class = class_names[predicted_class_idx]
        else:
            predicted_class = "Healthy" if predicted_class_idx == 1 else "Osteoporosis"

        # Select correct layer for Grad-CAM
        layer_name = "top_conv" if model == "efficientnet" else "block5_conv4"

        # Generate Grad-CAM heatmap
        heatmap = grad_cam(model_instance, img_array, target_size, layer_name)
        overlay_img = overlay_gradcam(img_bgr, heatmap)

        # Encode Grad-CAM image
        _, buffer = cv2.imencode(".jpg", overlay_img)
        overlay_base64 = base64.b64encode(buffer).decode()

        return JSONResponse(content={
            "prediction": predicted_class,
            "confidence": confidence_score,
            "gradcam_image": overlay_base64
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
# Add this to your FastAPI backend

@app.delete("/clear-reports/")
async def clear_reports():
    try:
        reports_collection.delete_many({})  # Delete all reports
        return JSONResponse(status_code=200, content={"message": "All reports cleared successfully"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Function to check if the image is a knee X-ray using Roboflow
async def is_knee_xray(image_bytes: bytes) -> bool:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result = CLIENT.infer(image, model_id="knee-detector/1")
        return bool(result['predictions'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting knee X-ray: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
