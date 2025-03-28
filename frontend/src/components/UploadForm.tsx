/* eslint-disable @next/next/no-img-element */
"use client";

import { useState, useCallback, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios";
import { toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

const ColorLegend = () => {
  const colors = [
    { color: "#0000FF", label: "Low Activation" },
    { color: "#00FFFF", label: "Moderate Activation" },
    { color: "#00FF00", label: "High Activation" },
    { color: "#FFFF00", label: "Very High Activation" },
    { color: "#FF0000", label: "Strongest Activation" },
  ];

  return (
    <div className="p-4 rounded-lg bg-gray-100 shadow-md">
      <h3 className="text-lg font-semibold mb-2">Grad-CAM Guide</h3>
      <div className="flex flex-col gap-2">
        {colors.map((item, index) => (
          <div key={index} className="flex items-center gap-2">
            <div
              className="w-6 h-6 rounded"
              style={{ backgroundColor: item.color }}
            ></div>
            <span className="text-sm">{item.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default function UploadForm() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<string>("");
  const [confidenceScore, setConfidenceScore] = useState<string>("");
  const [gradCamImage, setGradCamImage] = useState<string>("");
  const [uploadedImageUrl, setUploadedImageUrl] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [selectedModel, setSelectedModel] = useState<string>("binary_vgg19");

  useEffect(() => {
    if (selectedFile) {
      const objectUrl = URL.createObjectURL(selectedFile);
      setUploadedImageUrl(objectUrl);

      return () => URL.revokeObjectURL(objectUrl);
    }
  }, [selectedFile]);

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

  const handleUpload = async () => {
    if (!selectedFile) {
      toast.warning("Please select an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("model", selectedModel);

    const headers = {
      "Content-Type": "multipart/form-data",
    };

    setLoading(true);
    setGradCamImage(""); // Reset Grad-CAM image when a new upload starts

    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/upload/",
        formData,
        { headers }
      );

      setPrediction(response.data.prediction);
      setConfidenceScore(response.data.confidence);

      if (response.data.gradcam_image) {
        setGradCamImage(
          `data:image/jpeg;base64,${response.data.gradcam_image}`
        );
      } else {
        console.error("Grad-CAM image not found in the response.");
      }

      toast.success("Image uploaded successfully!");
    } catch (error: unknown) {
      if (axios.isAxiosError(error)) {
        toast.error(error.response?.data?.error || "Error processing image.");
      } else {
        toast.error("An unexpected error occurred.");
      }
      console.error("Error uploading image:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen bg-gradient-to-r from-blue-500 to-teal-400 p-8 flex items-center justify-center">
      <div
        className="absolute inset-0 bg-cover bg-center opacity-30"
        style={{ backgroundImage: 'url("/background.jpg")' }}
      ></div>
      <div className="relative z-10 container mx-auto p-10 bg-white shadow-xl rounded-lg max-w-6xl ">
        <div className="relative z-10 flex  p-10 flex-col items-center justify-center w-full max-w-6xl ">
          <div className="bg-white shadow-md rounded-lg p-10 w-full">
            <h1 className="text-3xl font-bold mb-6 text-center">
              Knee Osteoporosis Detection
            </h1>

            {/* Model Selection */}
            <div className="mb-6">
              <label className="block mb-2 font-semibold text-lg">
                Select Model:
              </label>
              <select
                className="w-full p-3 border border-gray-300 rounded-md"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
              >
                <option value="binary_vgg19">Binary - VGG19</option>
                <option value="efficientnet">Binary - EfficientNet</option>
                <option value="multiclass_vgg19">Multiclass - VGG19</option>
              </select>
            </div>

            {/* Dropzone */}
            <div
              {...getRootProps()}
              className={`border-2 border-dashed p-8 rounded-lg text-center cursor-pointer transition-all ${
                isDragActive
                  ? "border-blue-500 bg-blue-50"
                  : "border-gray-300 bg-gray-100"
              }`}
            >
              <input {...getInputProps()} />
              {selectedFile ? (
                <p className="text-gray-700">
                  Selected File: {selectedFile.name}
                </p>
              ) : (
                <p className="text-gray-500">
                  Drag & drop an image here, or click to select one
                </p>
              )}
            </div>

            {/* Upload Button */}
            <button
              onClick={handleUpload}
              disabled={loading}
              className={`w-full mt-4 py-3 rounded-md transition ${
                loading
                  ? "bg-gray-400 cursor-not-allowed"
                  : "bg-blue-600 text-white hover:bg-blue-700"
              } text-lg`}
            >
              {loading ? "Processing..." : "Upload & Predict"}
            </button>

            {/* Prediction & Confidence Score */}
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

            {/* Uploaded Image */}
            {uploadedImageUrl && (
              <div className="mt-6 flex flex-col items-center">
                <h3 className="text-xl font-medium">Uploaded Image</h3>
                <img
                  src={uploadedImageUrl}
                  alt="Uploaded"
                  className={`mt-4 rounded-md shadow-md w-[200px] h-[200px] object-cover transition-all duration-1000 ease-in-out transform ${
                    loading ? "scale-95 opacity-50" : "scale-100 opacity-100"
                  }`}
                />
                {loading && (
                  <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center rounded-md">
                    <p className="text-white text-lg font-bold">
                      Processing...
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Grad-CAM and Color Guide */}
            {gradCamImage && (
              <div className="mt-6 flex justify-center items-center gap-6">
                <div className="flex flex-col items-center">
                  <h3 className="text-xl font-medium mb-2">Grad-CAM Heatmap</h3>
                  <img
                    src={gradCamImage}
                    alt="Grad-CAM Overlay"
                    className="rounded-md shadow-md w-[220px] h-[220px] object-cover"
                  />
                </div>
                <div className="self-center">
                  <ColorLegend />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
