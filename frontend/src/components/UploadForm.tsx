"use client";

import { useState, useRef, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios";

export default function UploadForm() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<string>("");
  const [confidenceScore, setConfidenceScore] = useState<string>("");
  const [gradCamImage, setGradCamImage] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [selectedModel, setSelectedModel] = useState<string>("vgg19_binary"); // default model

  const fileInputRef = useRef<HTMLInputElement>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setSelectedFile(acceptedFiles[0]);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [] },
    multiple: false,
  });

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert("Please select an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("model_name", selectedModel); // Send selected model

    setLoading(true);
    try {
      const response = await axios.post(
        "http://127.0.0.1:8001/upload/",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );

      setPrediction(response.data.prediction);
      setConfidenceScore(response.data.confidence);
      if (response.data.gradcam_image) {
        setGradCamImage(`data:image/jpeg;base64,${response.data.gradcam_image}`);
      } else {
        console.error("Grad-CAM image not found in the response.");
      }
    } catch (error) {
      console.error("Error uploading image:", error);
      alert("Error uploading image. Check console for details.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen bg-gray-100 p-16 flex items-center justify-center">
      {/* Background Image */}
      <div
        className="absolute inset-0 bg-cover bg-center opacity-40"
        style={{ backgroundImage: 'url("/background.jpg")' }}
      ></div>

      <div className="relative z-10 flex flex-col items-center justify-center w-full max-w-4xl">
        <div className="bg-white shadow-md rounded-lg p-20 w-full">
          <h1 className="text-4xl font-bold mb-6 text-center">Knee Osteoporosis Detection</h1>

          {/* Model Dropdown */}
          <div className="mb-6">
            <label className="block mb-2 font-semibold text-lg">Select Model:</label>
            <select
              className="w-full p-3 border border-gray-300 rounded-md"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              <option value="vgg19_binary">VGG19 Binary Classifier</option>
              <option value="efficientnet_binary">EfficientNet Binary Classifier</option>
              <option value="vgg19_multiclass">VGG19 Multi-Class Classifier</option>
              {/* Add more models here if needed */}
            </select>
          </div>

          {/* Dropzone */}
          <div
            {...getRootProps()}
            className={`border-2 border-dashed p-8 rounded-lg text-center cursor-pointer transition-all ${
              isDragActive ? "border-blue-500 bg-blue-50" : "border-gray-300 bg-gray-100"
            }`}
          >
            <input {...getInputProps()} />
            {selectedFile ? (
              <p className="text-gray-700">Selected File: {selectedFile.name}</p>
            ) : (
              <p className="text-gray-500">Drag & drop an image here, or click to select one</p>
            )}
          </div>

          <button
            onClick={triggerFileInput}
            className="w-full mt-4 py-3 px-6 bg-gray-300 hover:bg-gray-400 text-gray-800 rounded-md text-lg"
          >
            Browse Files
          </button>

          <input
            type="file"
            ref={fileInputRef}
            onChange={(e) => e.target.files && setSelectedFile(e.target.files[0])}
            accept="image/*"
            className="hidden"
          />

          <button
            onClick={handleUpload}
            disabled={loading}
            className={`w-full mt-4 py-3 rounded-md transition ${
              loading ? "bg-gray-400 cursor-not-allowed" : "bg-blue-600 text-white hover:bg-blue-700"
            } text-lg`}
          >
            {loading ? "Processing..." : "Upload & Predict"}
          </button>

          {/* Results */}
          {prediction && (
            <div className="mt-6 text-center">
              <h2 className="text-2xl font-semibold">Prediction:</h2>
              <p className="text-blue-600 text-xl">{prediction}</p>
            </div>
          )}

          {confidenceScore && (
            <div className="mt-4 text-center">
              <h2 className="text-2xl font-semibold">Confidence Score:</h2>
              <p className="text-gray-700 text-xl">{confidenceScore}</p>
            </div>
          )}

          {gradCamImage && (
            <div className="mt-6">
              <h3 className="text-xl font-medium text-center">Grad-CAM Heatmap</h3>
              <img
                src={gradCamImage}
                alt="Grad-CAM Overlay"
                className="mt-4 rounded-md shadow-md mx-auto max-w-full"
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
