/* eslint-disable @next/next/no-img-element */
"use client";

import { useState, useEffect } from "react";
import axios from "axios";
import { jsPDF } from "jspdf";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { VisuallyHidden } from "@radix-ui/react-visually-hidden";
import { toast, ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
export default function ReportPage() {
  const [patientDetails, setPatientDetails] = useState({
    name: "",
    id: "",
    age: "",
    gender: "",
  });

  const [selectedModel, setSelectedModel] = useState("binary_vgg19");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<string>("");
  const [confidenceScore, setConfidenceScore] = useState<string>("");
  const [gradCamImage, setGradCamImage] = useState<string>("");
  const [doctorNotes, setDoctorNotes] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [isReportGenerating, setIsReportGenerating] = useState(false);
  const [reportDialogOpen, setReportDialogOpen] = useState(false);

  const [isInitialLoad, setIsInitialLoad] = useState(true); // New state to track initial load

  useEffect(() => {
    if (isInitialLoad) {
      setIsInitialLoad(false); // After the first load, stop loading from localStorage
      return;
    }
    // Only load from localStorage if it's not the initial load
    const savedPatientDetails = localStorage.getItem("patientDetails");
    const savedSelectedModel = localStorage.getItem("selectedModel");
    const savedUploadedImage = localStorage.getItem("uploadedImage");
    const savedPrediction = localStorage.getItem("prediction");
    const savedConfidenceScore = localStorage.getItem("confidenceScore");
    const savedGradCamImage = localStorage.getItem("gradCamImage");
    const savedDoctorNotes = localStorage.getItem("doctorNotes");

    if (savedPatientDetails) {
      setPatientDetails(JSON.parse(savedPatientDetails));
    }
    if (savedSelectedModel) {
      setSelectedModel(savedSelectedModel);
    }
    if (savedUploadedImage) {
      setUploadedImage(savedUploadedImage);
    }
    if (savedPrediction) {
      setPrediction(savedPrediction);
    }
    if (savedConfidenceScore) {
      setConfidenceScore(savedConfidenceScore);
    }
    if (savedGradCamImage) {
      setGradCamImage(savedGradCamImage);
    }
    if (savedDoctorNotes) {
      setDoctorNotes(savedDoctorNotes);
    }
  }, [isInitialLoad]);

  // Effect hooks to save state to localStorage
  useEffect(() => {
    if (!isInitialLoad) {
      localStorage.setItem("patientDetails", JSON.stringify(patientDetails));
      localStorage.setItem("selectedModel", selectedModel);
      if (uploadedImage) localStorage.setItem("uploadedImage", uploadedImage);
      if (prediction) localStorage.setItem("prediction", prediction);
      if (confidenceScore)
        localStorage.setItem("confidenceScore", confidenceScore);
      if (gradCamImage) localStorage.setItem("gradCamImage", gradCamImage);
      if (doctorNotes) localStorage.setItem("doctorNotes", doctorNotes);
    }
  }, [
    patientDetails,
    selectedModel,
    uploadedImage,
    prediction,
    confidenceScore,
    gradCamImage,
    doctorNotes,
    isInitialLoad,
  ]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setUploadedImage(URL.createObjectURL(file)); // Show preview of the uploaded image
    }
  };
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

  const handleUpload = async () => {
    if (!selectedFile) {
      toast.error("Please upload an X-ray image.");
      return;
    }
    setPrediction("");
    setConfidenceScore("");
    setGradCamImage("");
    setLoading(true);
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("model", selectedModel);

    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/upload/",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );

      setPrediction(response.data.prediction);
      setConfidenceScore(response.data.confidence);
      setGradCamImage(`data:image/jpeg;base64,${response.data.gradcam_image}`);
    } catch (error) {
      console.error("Error uploading image:", error);
      toast.error("Failed to analyze image.");
    } finally {
      setLoading(false);
    }
  };

  // Clear all stored data function
  const clearAllData = () => {
    // Remove all localStorage items
    localStorage.removeItem("patientDetails");
    localStorage.removeItem("selectedModel");
    localStorage.removeItem("uploadedImage");
    localStorage.removeItem("prediction");
    localStorage.removeItem("confidenceScore");
    localStorage.removeItem("gradCamImage");
    localStorage.removeItem("doctorNotes");

    // Reset all states
    setPatientDetails({
      name: "",
      id: "",
      age: "",
      gender: "",
    });
    setSelectedModel("binary_vgg19");
    setSelectedFile(null);
    setUploadedImage(null);
    setPrediction("");
    setConfidenceScore("");
    setGradCamImage("");
    setDoctorNotes("");
  };

  const generateReport = async () => {
    // Validate patient details
    if (!patientDetails.name) {
      toast.error("Please enter patient name before generating the report.");
      return;
    }

    // Open the dialog and start generating
    setReportDialogOpen(true);
    setIsReportGenerating(true);

    // Prepare the report data
    sendReportToDB();

    // Simulate report generation process
    await new Promise((resolve) => setTimeout(resolve, 2000));

    // Stop generating animation
    setIsReportGenerating(false);
  };
  const sendReportToDB = async () => {
    try {
      const doc = await formatReport();
      console.log("Doc:", doc);

      const pdfBlob = doc.output("blob"); // Get the PDF Blob
      console.log("Blob:", pdfBlob);
      if (!(pdfBlob instanceof Blob)) {
        console.error("pdfBlob is not a valid Blob:", pdfBlob);
      }

      const formData = new FormData();
      formData.append("patient_id", patientDetails.id);
      formData.append("patient_name", patientDetails.name);
      formData.append("model_used", selectedModel);
      formData.append("prediction", prediction);
      formData.append("confidence_score", confidenceScore);

      // Append the Blob directly as a file
      formData.append("report_pdf", pdfBlob, "report.pdf"); // The third argument is the filename

      // Log the form data entries
      console.log("Form Data Entries:");
      formData.forEach((value, key) => {
        console.log(key, value);
      });

      // Send the form data to the backend
      const response = await axios.post(
        "http://127.0.0.1:8000/save-report/",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      console.log("Response:", response);

      // Handle successful report saving
      if (response.status === 200) {
        toast.success("Report saved successfully!");
      }
    } catch (error) {
      console.error("Error saving report:", error);
      toast.error("Failed to save report.");
    } finally {
      setIsReportGenerating(false);
    }
  };

  const formatReport = async () => {
    const doc = new jsPDF({
      orientation: "portrait",
      unit: "mm",
      format: "a4",
    });

    const pageWidth = doc.internal.pageSize.width;
    const guideImage = "./image.png";

    // Add Osteo-Vision header
    doc.setFontSize(20);
    doc.setFont("helvetica", "bold");
    doc.text("Osteo-Vision", pageWidth / 2, 15, { align: "center" });

    // Report Title
    doc.setFontSize(16);
    doc.text("Knee Osteoporosis Analysis Report", pageWidth / 2, 25, {
      align: "center",
    });

    // Patient Details
    doc.setFontSize(10);
    doc.setFont("helvetica", "normal");
    const dateTime = new Date().toLocaleString();
    doc.text(`Generated on: ${dateTime}`, 20, 40);
    doc.text(`Patient Name: ${patientDetails.name}`, 20, 50);
    doc.text(`Patient ID: ${patientDetails.id}`, 20, 60);
    doc.text(`Age: ${patientDetails.age}`, 20, 70);
    doc.text(`Gender: ${patientDetails.gender}`, 20, 80);
    doc.text(`Model Used: ${selectedModel}`, 20, 90);
    doc.text(`Prediction: ${prediction}`, 20, 100);
    doc.text(`Confidence Score: ${confidenceScore}`, 20, 110);

    // Add uploaded X-ray image
    if (uploadedImage) {
      doc.addPage();
      doc.text("Original X-ray Image", pageWidth / 2, 15, { align: "center" });
      doc.addImage(uploadedImage, "JPEG", 50, 30, 110, 110);
    }

    // Add Grad-CAM image
    if (gradCamImage) {
      doc.addPage();
      doc.text("Grad-CAM Heatmap", pageWidth / 2, 15, { align: "center" });
      doc.addImage(gradCamImage, "JPEG", 50, 30, 110, 110);
      doc.addImage(guideImage, "PNG", 85, 150, 40, 40);
    }

    // Add Doctor's Notes
    doc.addPage();
    doc.text("Doctor's Notes:", 20, 20);
    doc.text(doctorNotes, 20, 30, { maxWidth: 170 });

    return doc;
  };
  const confirmReportDownload = async () => {
    const doc = await formatReport();

    // Save with patient name
    const sanitizedFileName = patientDetails.name
      .replace(/[^a-z0-9]/gi, "_")
      .toLowerCase();
    doc.save(`Osteo_Vision_Report_${sanitizedFileName}.pdf`);

    // Close the dialog
    setReportDialogOpen(false);
  };

  const cancelReportDownload = () => {
    setReportDialogOpen(false);
  };

  return (
    <div className="relative min-h-screen bg-gradient-to-r from-blue-500 to-teal-400 p-8 flex items-center justify-center">
      <div
        className="absolute inset-0 bg-cover bg-center opacity-40"
        style={{ backgroundImage: 'url("/background.jpg")' }}
      ></div>

      <div className="relative z-10 flex flex-col items-center justify-center w-full max-w-6xl ">
        <div className="bg-white shadow-lg rounded-xl p-12 w-full ">
          <h1 className="text-3xl font-bold mb-6 text-center text-gray-800">
            Generate Medical Report
          </h1>

          {/* Clear All Data Button */}
          <button
            onClick={clearAllData}
            className="absolute top-12 right-12 bg-red-500 hover:bg-red-600 text-white font-semibold py-2 px-4 rounded-xl shadow-md transform hover:scale-10 transition duration-300"
          >
            Clear
          </button>

          {/* Patient Details */}
          <div className="space-y-4 mb-4">
            <input
              type="text"
              placeholder="Patient Name"
              value={patientDetails.name}
              onChange={(e) =>
                setPatientDetails({ ...patientDetails, name: e.target.value })
              }
              className="w-full p-3 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <input
              type="text"
              placeholder="Patient ID"
              value={patientDetails.id}
              onChange={(e) =>
                setPatientDetails({ ...patientDetails, id: e.target.value })
              }
              className="w-full p-3 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <input
              type="number"
              placeholder="Age"
              value={patientDetails.age}
              onChange={(e) =>
                setPatientDetails({ ...patientDetails, age: e.target.value })
              }
              className="w-full p-3 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <select
              value={patientDetails.gender}
              onChange={(e) =>
                setPatientDetails({ ...patientDetails, gender: e.target.value })
              }
              className="w-full p-3 mb-4 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">Select Gender</option>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
              <option value="Other">Other</option>
            </select>
          </div>

          {/* Model Selection */}
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="w-full p-3 mb-4 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="binary_vgg19">Binary - VGG19</option>
            <option value="efficientnet">Binary - EfficientNet</option>
            <option value="multiclass_vgg19">Multiclass - VGG19</option>
          </select>

          {/* File Upload */}
          <div className="mb-6">
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              id="file-upload"
              className="hidden"
            />
            <label
              htmlFor="file-upload"
              className="cursor-pointer inline-block w-full text-center py-3 px-6 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-md shadow-md transition duration-300 transform hover:scale-10"
            >
              Select X-ray Image
            </label>
            {uploadedImage && (
              <div className="mt-6 flex flex-col items-center">
                <img
                  src={uploadedImage}
                  alt="Uploaded X-ray"
                  className={`mt-4 rounded-md shadow-md w-[250px] h-[250px] object-cover transition-all duration-1000 ease-in-out transform ${
                    loading ? "scale-95 opacity-50" : "scale-100 opacity-100"
                  }`}
                />
                {loading && (
                  <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center rounded-md">
                    <div className="w-12 h-12 border-4 border-white border-t-transparent rounded-full animate-spin"></div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Upload Button */}
          <button
            onClick={handleUpload}
            disabled={loading}
            className="w-full text-center py-3 px-6 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-md shadow-md text-base transform hover:scale-10 transition duration-300"
          >
            {loading ? "Processing..." : "Upload & Analyze"}
          </button>

          {/* Prediction & Confidence */}
          {prediction && (
            <div className="mt-6 text-center">
              <h2 className="text-2xl font-semibold">Prediction:</h2>
              <p className="text-blue-600 text-lg">{prediction}</p>
            </div>
          )}
          {confidenceScore && (
            <div className="mt-3 text-center">
              <h2 className="text-2xl font-semibold">Confidence Score:</h2>
              <p className="text-gray-700 text-lg">{confidenceScore}</p>
            </div>
          )}

          {/* Grad-CAM and Color Guide */}
          {gradCamImage && (
            <div className="mt-6 flex justify-center items-center gap-8">
              <div className="flex flex-col items-center">
                <h3 className="text-2xl font-medium mb-3">Grad-CAM Heatmap</h3>
                <img
                  src={gradCamImage}
                  alt="Grad-CAM Overlay"
                  className="rounded-md shadow-md w-[240px] h-[240px] object-cover"
                />
              </div>
              <div className="flex items-center gap-8">
                <ColorLegend />
              </div>
            </div>
          )}

          {/* Doctor Notes */}
          <textarea
            placeholder="Doctor's Notes"
            value={doctorNotes}
            onChange={(e) => setDoctorNotes(e.target.value)}
            className="w-full p-4 mt-5 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 text-base"
          ></textarea>

          {/* Generate Report Button */}
          <button
            onClick={generateReport}
            className="w-full py-3 mt-5 bg-green-600 hover:bg-green-700 text-white font-semibold rounded-md shadow-md text-base transform hover:scale-10 transition duration-300"
          >
            Generate Report
          </button>
        </div>
      </div>

      {/* Report Generation Dialog */}
      <Dialog open={reportDialogOpen} onOpenChange={setReportDialogOpen}>
        <DialogContent>
          {isReportGenerating ? (
            <div className="flex flex-col items-center justify-center p-6">
              {/* Hidden title for accessibility */}
              <VisuallyHidden>
                <DialogTitle>Generating Report</DialogTitle>
              </VisuallyHidden>
              <div className="animate-spin rounded-full h-12 w-12 border-t-4 border-blue-500 mb-4"></div>
              <p className="text-lg font-semibold">Generating your report...</p>
            </div>
          ) : (
            <>
              <DialogHeader>
                <DialogTitle>Report Generation Complete</DialogTitle>
                <DialogDescription>
                  Your medical report is ready to be downloaded.
                </DialogDescription>
              </DialogHeader>
              <DialogFooter>
                <Button variant="outline" onClick={cancelReportDownload}>
                  Cancel
                </Button>
                <Button variant="outline" onClick={confirmReportDownload}>
                  Download Report
                </Button>
              </DialogFooter>
            </>
          )}
        </DialogContent>
      </Dialog>

      <ToastContainer />
    </div>
  );
}
