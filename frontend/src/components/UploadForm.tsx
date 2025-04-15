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
    { color: "bg-blue-600", label: "Low Activation" },
    { color: "bg-cyan-400", label: "Moderate Activation" },
    { color: "bg-green-500", label: "High Activation" },
    { color: "bg-yellow-400", label: "Very High Activation" },
    { color: "bg-red-500", label: "Strongest Activation" },
  ];

  return (
    <div className="p-4 rounded-lg bg-gray-50 border border-gray-200 shadow-sm">
      <h3 className="text-lg font-medium text-gray-800 mb-3">Grad-CAM Guide</h3>
      <div className="space-y-2">
        {colors.map((item, index) => (
          <div key={index} className="flex items-center gap-3">
            <div className={`w-5 h-5 rounded ${item.color}`}></div>
            <span className="text-sm text-gray-600">{item.label}</span>
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
  const [showVisualizations, setShowVisualizations] = useState<boolean>(false);
  const [visualizationsLoading, setVisualizationsLoading] = useState<boolean>(false);
  const [visualizationsLoaded, setVisualizationsLoaded] = useState<boolean>(false);

  useEffect(() => {
    if (selectedFile) {
      const objectUrl = URL.createObjectURL(selectedFile);
      setUploadedImageUrl(objectUrl);
      return () => URL.revokeObjectURL(objectUrl);
    }
  }, [selectedFile]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      setSelectedFile(file);
      setPrediction("");
      setConfidenceScore("");
      setGradCamImage("");
      setLimeImage("");
      setShowVisualizations(false);
      setVisualizationsLoaded(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [] },
    multiple: false,
  });

  const loadVisualizations = async (formData: FormData) => {
    if (visualizationsLoaded) return;
    
    setVisualizationsLoading(true);
    
    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/visualize/",
        formData
      );

      if (response.data.gradcam_image) {
        setGradCamImage(`data:image/jpeg;base64,${response.data.gradcam_image}`);
      }

      if (response.data.lime_image) {
        setLimeImage(`data:image/jpeg;base64,${response.data.lime_image}`);
      }
      
      setVisualizationsLoaded(true);
    } catch (error) {
      console.error("Error loading visualizations:", error);
      toast.error("Failed to load visualizations");
    } finally {
      setVisualizationsLoading(false);
    }
  };

  const handleViewVisualizations = () => {
    if (!visualizationsLoaded && selectedFile) {
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("model", selectedModel);
      loadVisualizations(formData);
    }
    
    setShowVisualizations(true);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      toast.warning("Please select an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("model", selectedModel);

    setLoading(true);
    setProgressValue(0);
    setPrediction("");
    setConfidenceScore("");
    setGradCamImage("");
    setLimeImage("");
    setShowVisualizations(false);
    setVisualizationsLoaded(false);

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
        "http://127.0.0.1:8000/predict/",
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      setPrediction(response.data.prediction);
      setConfidenceScore(response.data.confidence);
      toast.success("Analysis completed successfully!");

      loadVisualizations(formData);
      
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
    <div className="pt-[60px]">
    <div className="relative min-h-screen bg-gradient-to-r from-blue-500 to-teal-400 flex items-center justify-center p-6">
      {/* Background image */}
      <div
        className="absolute inset-0 bg-cover bg-center opacity-20 z-0"
        style={{ backgroundImage: 'url("/background.jpg")' }}
      />

<div className="relative z-10 bg-white rounded-2xl shadow-2xl p-10 w-full max-w-6xl">
        {/* Main Card */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
          <div className="p-6 md:p-8 space-y-6">
            {/* Header */}
            <div className="text-center">
              <h1 className="text-2xl md:text-3xl font-bold text-gray-800">Knee Osteoporosis Detection</h1>
              <p className="text-gray-600 mt-2">AI-powered medical imaging analysis</p>
            </div>

            {/* Model Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">AI Model</label>
              <select
                className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
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
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Knee X-ray Image</label>
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                  isDragActive ? "border-blue-500 bg-blue-50" : "border-gray-300 hover:border-blue-400 bg-gray-50"
                }`}
              >
                <input {...getInputProps()} />
                <div className="flex flex-col items-center justify-center gap-2">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                  {selectedFile ? (
                    <p className="text-gray-700">{selectedFile.name}</p>
                  ) : (
                    <>
                      <p className="text-gray-600">Drag & drop an image here</p>
                      <p className="text-sm text-gray-500">or click to browse files</p>
                    </>
                  )}
                </div>
              </div>
            </div>

            {/* Upload Button */}
            <button
              onClick={handleUpload}
              disabled={loading || !selectedFile}
              className={`w-full py-3 px-6 rounded-lg font-medium text-white transition-colors ${
                loading || !selectedFile
                  ? "bg-gray-400 cursor-not-allowed"
                  : "bg-blue-600 hover:bg-blue-700"
              }`}
            >
              {loading ? (
                <div className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Analyzing...
                </div>
              ) : (
                "Analyze Image"
              )}
            </button>

            {/* Progress Bar */}
            {loading && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm text-gray-600">
                  <span>Processing image...</span>
                  <span>{progressValue}%</span>
                </div>
                <Progress value={progressValue} className="h-2" />
              </div>
            )}

            {/* Results Section */}
            {!loading && (prediction || confidenceScore) && (
              <div className="space-y-4">
                <h2 className="text-xl font-semibold text-gray-800">Analysis Results</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <h3 className="text-sm font-medium text-blue-800 mb-1">Prediction</h3>
                    <p className="text-2xl font-bold text-blue-900">{prediction}</p>
                  </div>
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <h3 className="text-sm font-medium text-blue-800 mb-1">Confidence Score</h3>
                    <p className="text-2xl font-bold text-blue-900">{confidenceScore}</p>
                  </div>
                </div>
              </div>
            )}

            {/* Uploaded Image Preview */}
            {uploadedImageUrl && (
              <div className="space-y-2">
                <h3 className="text-lg font-medium text-gray-800">Image Preview</h3>
                <div className="flex justify-center">
                  <div className="relative w-full max-w-xs aspect-square border border-gray-200 rounded-lg overflow-hidden">
                    <img
                      src={uploadedImageUrl}
                      alt="Uploaded X-ray"
                      className={`w-full h-full object-contain ${loading ? "opacity-70" : "opacity-100"}`}
                    />
                    {loading && (
                      <div className="absolute inset-0 bg-black bg-opacity-10 flex items-center justify-center">
                        <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Show Visualizations Button */}
            {!loading && prediction && !showVisualizations && (
              <div className="pt-4 text-center">
                <button
                  onClick={handleViewVisualizations}
                  className="px-6 py-3 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium transition-colors"
                >
                  View AI Explanations
                </button>
              </div>
            )}

            {/* Visualizations Section */}
            {showVisualizations && (
              <div className="space-y-6">
                <h2 className="text-xl font-semibold text-gray-800">AI Explanations</h2>
                
                {visualizationsLoading && (
                  <div className="space-y-2">
                    <div className="h-2.5 w-full bg-gray-200 rounded-full overflow-hidden">
                      <div className="h-full bg-blue-600 rounded-full animate-pulse" style={{ width: "70%" }}></div>
                    </div>
                    <p className="text-sm text-gray-500 text-center">Generating visual explanations...</p>
                  </div>
                )}

                {!visualizationsLoading && (
                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {gradCamImage && (
                      <div className="space-y-2">
                        <h3 className="text-lg font-medium text-gray-700">Grad-CAM Heatmap</h3>
                        <div className="border border-gray-200 rounded-lg overflow-hidden">
                          <img
                            src={gradCamImage}
                            alt="Grad-CAM Overlay"
                            className="w-full h-auto object-contain"
                          />
                        </div>
                      </div>
                    )}
                    
                    <div className="space-y-2">
                      <h3 className="text-lg font-medium text-gray-700">Interpretation Guide</h3>
                      <ColorLegend />
                    </div>
                    
                    {limeImage && (
                      <div className="space-y-2">
                        <h3 className="text-lg font-medium text-gray-700">LIME Visualization</h3>
                        <div className="border border-gray-200 rounded-lg overflow-hidden">
                          <img
                            src={limeImage}
                            alt="LIME Explanation"
                            className="w-full h-auto object-contain"
                          />
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
    </div>
  );
}