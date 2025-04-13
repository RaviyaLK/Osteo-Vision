/* eslint-disable @next/next/no-img-element */
"use client";

import { useState, useCallback, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios";
import { toast } from "react-toastify";
import { Progress } from "@/components/ui/progress";
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
  const [limeImage, setLimeImage] = useState<string>("");
  const [uploadedImageUrl, setUploadedImageUrl] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [progressValue, setProgressValue] = useState<number>(0);
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

    const headers = { "Content-Type": "multipart/form-data" };

    setLoading(true);
    setProgressValue(0);
    setPrediction("");
    setConfidenceScore("");
    setGradCamImage("");
    setLimeImage("");

    // Simulate progress bar
    let progress = 0;
    const interval = setInterval(() => {
      progress += 10;
      if (progress < 95) {
        setProgressValue(progress);
      } else {
        clearInterval(interval);
      }
    }, 200);

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

      if (response.data.lime_image) {
        setLimeImage(`data:image/jpeg;base64,${response.data.lime_image}`);
      } else {
        console.error("LIME image not found in the response.");
      }

      toast.success("Image uploaded successfully!");
    } catch (error: unknown) {
      if (axios.isAxiosError(error)) {
        toast.error(error.response?.data?.error || "Error processing image.");
      } else {
        toast.error("An unexpected error occurred.");
      }
    } finally {
      clearInterval(interval);
      setProgressValue(100);
      setTimeout(() => {
        setLoading(false);
        setProgressValue(0);
      }, 1000);
    }
  };

  return (
    <div className="relative min-h-screen bg-gradient-to-r from-blue-500 to-teal-400 flex items-center justify-center p-6">
      <div
        className="absolute inset-0 bg-cover bg-center opacity-20 z-0"
        style={{ backgroundImage: 'url("/background.jpg")' }}
      />
      <div className="relative z-10 bg-white rounded-2xl shadow-2xl p-10 w-full max-w-6xl">
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
            <option value="ensemble">Binary - Ensemble</option>
            <option value="multiclass_vgg19">Multiclass - VGG19</option>
          </select>
        </div>

        {/* Dropzone */}
        <div
          {...getRootProps()}
          className={`border-2 border-dashed p-8 rounded-lg text-center cursor-pointer transition ${
            isDragActive
              ? "border-blue-500 bg-blue-50"
              : "border-gray-300 bg-gray-100"
          }`}
        >
          <input {...getInputProps()} />
          {selectedFile ? (
            <p className="text-gray-700">Selected File: {selectedFile.name}</p>
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
          className={`w-full mt-4 py-3 rounded-md text-lg font-semibold transition ${
            loading
              ? "bg-gray-400 cursor-not-allowed"
              : "bg-blue-600 text-white hover:bg-blue-700"
          }`}
        >
          {loading ? "Processing..." : "Upload & Predict"}
        </button>

        {/* Progress Bar */}
        {loading && (
          <div className="mt-6">
            <Progress value={progressValue} className="h-4 w-full" />
          </div>
        )}

        {/* Results */}
        {!loading && prediction && (
          <div className="mt-6 text-center">
            <h2 className="text-2xl font-semibold">Prediction:</h2>
            <p className="text-blue-600 text-xl">{prediction}</p>
          </div>
        )}

        {!loading && confidenceScore && (
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
              className={`mt-4 rounded-md shadow-md w-[200px] h-[200px] object-cover ${
                loading ? "animate-pulse" : ""
              }`}
            />
          </div>
        )}

        {/* Visualizations */}
        {(gradCamImage || limeImage) && (
          <div className="mt-8 flex flex-col md:flex-row justify-center items-center gap-10">
            {gradCamImage && (
              <div className="flex flex-col items-center">
                <h3 className="text-xl font-medium mb-2">Grad-CAM Heatmap</h3>
                <img
                  src={gradCamImage}
                  alt="Grad-CAM Overlay"
                  className="rounded-md shadow-md w-[220px] h-[220px] object-cover"
                />
              </div>
            )}
            <div className="flex flex-col items-center">
              <h3 className="text-xl font-medium mb-2">Grad-CAM Guide</h3>
              <ColorLegend />
            </div>
            {limeImage && (
              <div className="flex flex-col items-center">
                <h3 className="text-xl font-medium mb-2">LIME Visualization</h3>
                <img
                  src={limeImage}
                  alt="LIME Explanation"
                  className="rounded-md shadow-md w-[220px] h-[220px] object-cover"
                />
              </div>
            )}
            
          </div>
        )}
      </div>
    </div>
  );
}
