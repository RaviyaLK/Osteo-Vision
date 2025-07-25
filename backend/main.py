import asyncio
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
from config import MONGO_URL, MODEL_NAME
from models.model_loader import load_model
from utils.gradcam import grad_cam, overlay_gradcam
from utils.lime import generate_lime_image
from utils.preprocessing import preprocess_image
from inference_sdk import InferenceHTTPClient
from config import ROBOFLOW_API_URL, ROBOFLOW_API_KEY
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

# CORS
origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Roboflow client
CLIENT = InferenceHTTPClient(api_url=ROBOFLOW_API_URL, api_key=ROBOFLOW_API_KEY)

# MongoDB setup
client = MongoClient(MONGO_URL)
db = client.reportsDB
reports_collection = db.reports

# Thread pool executor for parallel processing
executor = ThreadPoolExecutor(max_workers=4)

@tf.function(reduce_retracing=True)
def predict_tensorflow(model, input_tensor):
    return model(input_tensor, training=False)
def run_prediction(model, img_array, is_multiclass=False):
    prediction = model(img_array, training=False).numpy()
    if is_multiclass:
        predicted_class_idx = int(np.argmax(prediction))
        confidence_score = min(round(float(np.max(prediction)), 2), 0.99)
    else:
        predicted_class_idx = int(round(prediction[0][0]))
        confidence_score =round(float(prediction[0][0] if predicted_class_idx == 1 else 1 - prediction[0][0]), 2)
    return predicted_class_idx, confidence_score

class Report(BaseModel):
    patient_id: str
    patient_name: str
    model_used: str
    prediction: str
    confidence_score: float
    report_pdf: bytes

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
        pdf_bytes = await report_pdf.read()
        report_data = {
            "patient_id": patient_id,
            "patient_name": patient_name,
            "model_used": model_used,
            "prediction": prediction,
            "confidence_score": confidence_score,
            "report_pdf": pdf_bytes,
            "generated_on": datetime.now(),
        }
        result = reports_collection.insert_one(report_data)
        return JSONResponse(content={"message": "Report saved successfully!", "id": str(result.inserted_id)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving report: {str(e)}")

@app.get("/get-reports/")
async def get_reports():
    reports = list(reports_collection.find({}, {"report_pdf": 0}))
    for report in reports:
        report['_id'] = str(report['_id'])
    return reports

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


@app.post("/predict/")
async def predict_only(file: UploadFile = File(...), model: str = Form("binary_vgg19")):
    try:
        image_bytes = await file.read()
        if not image_bytes:
            return JSONResponse(status_code=400, content={"error": "Empty file uploaded"})

        is_knee = await is_knee_xray(image_bytes)
        if not is_knee:
            return JSONResponse(status_code=400, content={"error": "Uploaded image is not a knee X-ray"})

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if model == "ensemble":
            model_vgg = load_model("binary_vgg19")
            model_eff = load_model("efficientnet")

            img_vgg, target_size_vgg, image_resized_vgg = preprocess_image(image, "binary_vgg19")
            img_eff, target_size_eff, image_resized_eff = preprocess_image(image, "efficientnet")

            loop = asyncio.get_event_loop()
            vgg_result, eff_result = await asyncio.gather(
                loop.run_in_executor(executor, run_prediction, model_vgg, img_vgg, False),
                loop.run_in_executor(executor, run_prediction, model_eff, img_eff, False)
            )

            vgg_class_idx, vgg_confidence = vgg_result
            eff_class_idx, eff_confidence = eff_result
            vgg_prob = vgg_confidence if vgg_class_idx == 1 else 1 - vgg_confidence
            eff_prob = eff_confidence if eff_class_idx == 1 else 1 - eff_confidence
            final_prob = 0.6 * vgg_prob + 0.4 * eff_prob

            predicted_class_idx = int(final_prob >= 0.5)
            predicted_class = "Healthy" if predicted_class_idx == 1 else "Osteoporosis"
            confidence_score = round(final_prob if predicted_class_idx == 1 else 1 - final_prob, 2)

        else:
            model_instance = load_model(model)
            img_array, _, _ = preprocess_image(image, model)

            is_multiclass = model == "multiclass_vgg19"
            loop = asyncio.get_event_loop()
            predicted_class_idx, confidence_score = await loop.run_in_executor(
                executor, run_prediction, model_instance, img_array, is_multiclass
            )

            if is_multiclass:
                class_names = ["Healthy", "Osteopenia", "Osteoporosis"]
                predicted_class = class_names[predicted_class_idx]
            else:
                predicted_class = "Healthy" if predicted_class_idx == 1 else "Osteoporosis"

        return JSONResponse(content={
            "prediction": predicted_class,
            "confidence": confidence_score
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/visualize/")
async def generate_visualizations(file: UploadFile = File(...), model: str = Form("binary_vgg19")):
    try:
        image_bytes = await file.read()
        if not image_bytes:
            return JSONResponse(status_code=400, content={"error": "Empty file uploaded"})

        is_knee = await is_knee_xray(image_bytes)
        if not is_knee:
            return JSONResponse(status_code=400, content={"error": "Uploaded image is not a knee X-ray"})

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if model == "ensemble":
            model_instance = load_model("binary_vgg19")
            img_array, target_size, image_resized = preprocess_image(image, "binary_vgg19")
            layer_name = "block5_conv4"
        else:
            model_instance = load_model(model)
            img_array, target_size, image_resized = preprocess_image(image, model)
            layer_name = "top_conv" if model == "efficientnet" else "block5_conv4"

        # Grad-CAM
        img_bgr = cv2.cvtColor(np.array(image_resized), cv2.COLOR_RGB2BGR)
        heatmap = grad_cam(model_instance, img_array, target_size, layer_name)
        overlay_img = overlay_gradcam(img_bgr, heatmap)
        _, buffer = cv2.imencode(".jpg", overlay_img)
        overlay_base64 = base64.b64encode(buffer).decode()

        # LIME
        lime_image_b64 = generate_lime_image(model_instance, img_array, image_resized, model)

        return JSONResponse(content={
            "gradcam_image": overlay_base64,
            "lime_image": lime_image_b64
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.delete("/clear-reports/")
async def clear_reports():
    try:
        reports_collection.delete_many({})
        return JSONResponse(status_code=200, content={"message": "All reports cleared successfully"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

async def is_knee_xray(image_bytes: bytes) -> bool:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result = CLIENT.infer(image, model_id=MODEL_NAME)
        return bool(result['predictions'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting knee X-ray: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
