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
  const [limeImage, setlimeImage] = useState<string>("");
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
    const savedlimeImage = localStorage.getItem("limeImage");
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
    if (savedlimeImage) {
      setlimeImage(savedlimeImage);
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
      if (limeImage) localStorage.setItem("limeImage", limeImage);
      if (doctorNotes) localStorage.setItem("doctorNotes", doctorNotes);
    }
  }, [
    patientDetails,
    selectedModel,
    uploadedImage,
    prediction,
    confidenceScore,
    gradCamImage,
    limeImage,
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
    setlimeImage("");
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
      setlimeImage(`data:image/jpeg;base64,${response.data.lime_image}`);
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
    setlimeImage("");
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
    
    // Constants for layout
    const pageWidth = doc.internal.pageSize.width;
    const pageHeight = doc.internal.pageSize.height; 
    const margin = 20;
    const guideImage = "./image.png";
    
    // Add custom header with logo
    const addHeader = () => {
      // Header line
      doc.setDrawColor(0, 102, 204);
      doc.setLineWidth(0.5);
      doc.line(margin, margin, pageWidth - margin, margin);
      
      // Osteo-Vision header
      doc.setFontSize(22);
      doc.setFont("helvetica", "bold");
      doc.setTextColor(0, 102, 204);
      doc.text("Osteo-Vision", margin, margin - 5);
      
      // Subtitle
      doc.setFontSize(10);
      doc.setFont("helvetica", "italic");
      doc.setTextColor(100, 100, 100);
      doc.text("Advanced Osteoporosis Detection & Analysis", margin, margin + 5);
      
      // Bottom line
      doc.line(margin, pageHeight - margin, pageWidth - margin, pageHeight - margin);
      
      // Page number at bottom
      doc.setFontSize(8);
      doc.setFont("helvetica", "normal");
      doc.setTextColor(100, 100, 100);
      const pageInfo = `Page ${doc.internal.pages.length - 1}`;
      doc.text(pageInfo, pageWidth - margin, pageHeight - margin + 5, { align: "right" });
    };
    
    // Cover page
    addHeader();
    
    // Report Title
    doc.setFontSize(18);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(0, 0, 0);
    doc.text("KNEE OSTEOPOROSIS ANALYSIS REPORT", pageWidth / 2, 45, {
      align: "center",
    });
    
    // Add decorative element
    doc.setDrawColor(0, 102, 204);
    doc.setLineWidth(0.3);
    doc.line(pageWidth / 2 - 40, 48, pageWidth / 2 + 40, 48);
    
    // Patient Information Section
    doc.setFillColor(240, 240, 240);
    doc.rect(margin, 60, pageWidth - (margin * 2), 50, "F");
    
    doc.setFontSize(12);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(0, 0, 0);
    doc.text("PATIENT INFORMATION", margin + 5, 70);
    
    doc.setLineWidth(0.2);
    doc.setDrawColor(200, 200, 200);
    doc.line(margin + 5, 72, margin + 60, 72);
    
    // Patient Details
    doc.setFontSize(10);
    doc.setFont("helvetica", "normal");
    const dateTime = new Date().toLocaleString();
    
    const leftColX = margin + 5;
    const rightColX = pageWidth / 2 + 10;
    
    doc.text(`Patient Name:`, leftColX, 80);
    doc.text(`${patientDetails.name}`, leftColX + 30, 80);
    
    doc.text(`Patient ID:`, leftColX, 88);
    doc.text(`${patientDetails.id}`, leftColX + 30, 88);
    
    doc.text(`Age:`, leftColX, 96);
    doc.text(`${patientDetails.age}`, leftColX + 30, 96);
    
    doc.text(`Gender:`, rightColX, 80);
    doc.text(`${patientDetails.gender}`, rightColX + 30, 80);
    
    doc.text(`Report Date:`, rightColX, 88);
    doc.text(`${dateTime}`, rightColX + 30, 88);
    
    // Analysis Results Section
    doc.setFillColor(230, 240, 250);
    doc.rect(margin, 120, pageWidth - (margin * 2), 40, "F");
    
    doc.setFontSize(12);
    doc.setFont("helvetica", "bold");
    doc.text("ANALYSIS RESULTS", margin + 5, 130);
    
    doc.setLineWidth(0.2);
    doc.setDrawColor(0, 102, 204);
    doc.line(margin + 5, 132, margin + 50, 132);
    
    doc.setFontSize(10);
    doc.setFont("helvetica", "normal");
    
    doc.text(`Model Used:`, leftColX, 142);
    doc.text(`${selectedModel}`, leftColX + 30, 142);
    
    doc.text(`Prediction:`, leftColX, 150);
    doc.setFont("helvetica", "bold");
    doc.text(`${prediction}`, leftColX + 30, 150);
    doc.setFont("helvetica", "normal");
    
    doc.text(`Confidence Score:`, rightColX, 142);
    doc.text(`${confidenceScore}`, rightColX + 40, 142);
    
    // Add second page with original X-ray
    if (uploadedImage) {
      doc.addPage();
      addHeader();
      
      doc.setFontSize(14);
      doc.setFont("helvetica", "bold");
      doc.text("ORIGINAL X-RAY IMAGE", pageWidth / 2, 40, { align: "center" });
      
      // Add subheading
      doc.setFontSize(10);
      doc.setFont("helvetica", "italic");
      doc.text("Patient knee radiograph submitted for analysis", pageWidth / 2, 48, { align: "center" });
      
      // Add image with border
      doc.setDrawColor(200, 200, 200);
      doc.setLineWidth(0.5);
      const imageX = (pageWidth - 120) / 2;
      doc.rect(imageX - 1, 55 - 1, 122, 122);
      doc.addImage(uploadedImage, "JPEG", imageX, 55, 120, 120);
      
      // Add image description
      doc.setFontSize(9);
      doc.setFont("helvetica", "normal");
      doc.text("Fig. 1: Original knee radiograph", pageWidth / 2, 185, { align: "center" });
    }
    
    // Add third page with both Grad-CAM and LIME on same page
    if (gradCamImage || limeImage) {
      doc.addPage();
      addHeader();
      
      doc.setFontSize(14);
      doc.setFont("helvetica", "bold");
      doc.text("AI ANALYSIS VISUALIZATIONS", pageWidth / 2, 40, { align: "center" });
      
      // Add explanation
      doc.setFontSize(9);
      doc.setFont("helvetica", "italic");
      doc.text("The following images show areas analyzed by the AI model to reach its diagnosis", pageWidth / 2, 48, { align: "center" });
      
      if (gradCamImage && limeImage) {
        // Layout for both images
        const imgWidth = 80;
        const leftImgX = margin + 10;
        const rightImgX = pageWidth - margin - imgWidth - 10;
        const imgY = 60;
        
        // First image - Grad-CAM
        doc.setDrawColor(200, 200, 200);
        doc.setLineWidth(0.5);
        doc.rect(leftImgX - 1, imgY - 1, imgWidth + 2, imgWidth + 2);
        doc.addImage(gradCamImage, "JPEG", leftImgX, imgY, imgWidth, imgWidth);
        
        // Second image - LIME
        doc.rect(rightImgX - 1, imgY - 1, imgWidth + 2, imgWidth + 2);
        doc.addImage(limeImage, "JPEG", rightImgX, imgY, imgWidth, imgWidth);
        
        // Image captions
        doc.setFontSize(9);
        doc.setFont("helvetica", "normal");
        doc.text("Fig. 2: Grad-CAM Heatmap", leftImgX + imgWidth/2, imgY + imgWidth + 10, { align: "center" });
        doc.text("Fig. 3: LIME Explanations", rightImgX + imgWidth/2, imgY + imgWidth + 10, { align: "center" });
        
        // Add legend/guide image centered below
        const guideWidth = 50;
        const guideX = (pageWidth - guideWidth) / 2;
        doc.addImage(guideImage, "PNG", guideX, imgY + imgWidth + 20, guideWidth, guideWidth);
        
        // Legend title
        doc.setFontSize(9);
        doc.setFont("helvetica", "italic");
        doc.text("Interpretation Guide", pageWidth / 2, imgY + imgWidth + 80, { align: "center" });
        
        // Add explanation of the visualizations
        doc.setFontSize(9);
        doc.setFont("helvetica", "normal");
        const explanationText = 
          "Grad-CAM (Gradient-weighted Class Activation Mapping) highlights regions that influenced the AI's decision most significantly. " +
          "Red areas indicate regions of highest importance for the diagnosis. " +
          "LIME (Local Interpretable Model-agnostic Explanations) identifies specific features that contributed to the prediction. " +
          "These visualizations help clinicians understand how the AI reached its conclusion.";
        
        doc.text(explanationText, margin, imgY + imgWidth + 90, { 
          align: "left",
          maxWidth: pageWidth - (margin * 2)
        });
      } else if (gradCamImage) {
        // Only Grad-CAM available
        const imgWidth = 110;
        const imgX = (pageWidth - imgWidth) / 2;
        doc.addImage(gradCamImage, "JPEG", imgX, 60, imgWidth, imgWidth);
        doc.text("Fig. 2: Grad-CAM Heatmap", pageWidth / 2, 180, { align: "center" });
        doc.addImage(guideImage, "PNG", (pageWidth - 40) / 2, 190, 40, 40);
      } else if (limeImage) {
        // Only LIME available
        const imgWidth = 110;
        const imgX = (pageWidth - imgWidth) / 2;
        doc.addImage(limeImage, "JPEG", imgX, 60, imgWidth, imgWidth);
        doc.text("Fig. 2: LIME Explanations", pageWidth / 2, 180, { align: "center" });
      }
    }
    
    // Add Doctor's Notes
    doc.addPage();
    addHeader();
    
    doc.setFontSize(14);
    doc.setFont("helvetica", "bold");
    doc.text("CLINICAL ASSESSMENT", pageWidth / 2, 40, { align: "center" });
    
    // Add decorative element
    doc.setDrawColor(0, 102, 204);
    doc.setLineWidth(0.3);
    doc.line(pageWidth / 2 - 30, 43, pageWidth / 2 + 30, 43);
    
    // Section for doctor's notes with border
    doc.setDrawColor(220, 220, 220);
    doc.setLineWidth(0.5);
    doc.rect(margin, 50, pageWidth - (margin * 2), pageHeight - 100);
    
    doc.setFontSize(11);
    doc.setFont("helvetica", "bold");
    doc.text("Doctor's Notes:", margin + 5, 60);
    
    doc.setFontSize(10);
    doc.setFont("helvetica", "normal");
    doc.text(doctorNotes, margin + 5, 70, { maxWidth: pageWidth - (margin * 2) - 10 });
    
    // Add signature area at bottom
    doc.text("Physician's Signature: _________________________", margin + 5, pageHeight - 40);
    doc.text("Date: _________________", pageWidth - 70, pageHeight - 40);
    
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
    <div className="relative min-h-screen bg-linear-to-r from-blue-500 to-teal-400 p-8 flex items-center justify-center">
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
            className="absolute top-12 right-12 bg-red-500 hover:bg-red-600 text-white font-semibold py-2 px-4 rounded-xl shadow-md "
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
              className="w-full p-3 border border-gray-300 rounded-md shadow-xs focus:outline-hidden focus:ring-2 focus:ring-blue-500"
            />
            <input
              type="text"
              placeholder="Patient ID"
              value={patientDetails.id}
              onChange={(e) =>
                setPatientDetails({ ...patientDetails, id: e.target.value })
              }
              className="w-full p-3 border border-gray-300 rounded-md shadow-xs focus:outline-hidden focus:ring-2 focus:ring-blue-500"
            />
            <input
              type="number"
              placeholder="Age"
              value={patientDetails.age}
              onChange={(e) =>
                setPatientDetails({ ...patientDetails, age: e.target.value })
              }
              className="w-full p-3 border border-gray-300 rounded-md shadow-xs focus:outline-hidden focus:ring-2 focus:ring-blue-500"
            />
            <select
              value={patientDetails.gender}
              onChange={(e) =>
                setPatientDetails({ ...patientDetails, gender: e.target.value })
              }
              className="w-full p-3 mb-4 border border-gray-300 rounded-md shadow-xs focus:outline-hidden focus:ring-2 focus:ring-blue-500"
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
            className="w-full p-3 mb-4 border border-gray-300 rounded-md shadow-xs focus:outline-hidden focus:ring-2 focus:ring-blue-500"
          >
            <option value="binary_vgg19">Binary - VGG19</option>
            <option value="efficientnet">Binary - EfficientNet</option>
            <option value="ensemble">Binary - Ensemble</option>
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
              className="cursor-pointer inline-block w-full text-center py-3 px-6 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-md shadow-md transition duration-300"
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
              </div>
            )}
          </div>

          {/* Upload Button */}
          <button
            onClick={handleUpload}
            disabled={loading}
            className="w-full text-center py-3 px-6 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-md shadow-md text-base "
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
                  <h3 className="text-xl font-medium mb-2">
                    LIME Visualization
                  </h3>
                  <img
                    src={limeImage}
                    alt="LIME Explanation"
                    className="rounded-md shadow-md w-[220px] h-[220px] object-cover"
                  />
                </div>
              )}
            </div>
          )}

          {/* Doctor Notes */}
          <textarea
            placeholder="Doctor's Notes"
            value={doctorNotes}
            onChange={(e) => setDoctorNotes(e.target.value)}
            className="w-full p-4 mt-5 border border-gray-300 rounded-md shadow-xs focus:outline-hidden focus:ring-2 focus:ring-blue-500 text-base"
          ></textarea>

          {/* Generate Report Button */}
          <button
            onClick={generateReport}
            className="w-full py-3 mt-5 bg-green-600 hover:bg-green-700 text-white font-semibold rounded-md shadow-md text-base "
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
