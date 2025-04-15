/* eslint-disable @next/next/no-img-element */
"use client";

import { useState, useEffect } from "react";
import axios from "axios";
import { jsPDF } from "jspdf";
import { toast, ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import ReportGenerationDialog from "@/components/ReportGenerationDialog";

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
  const [visualizationsLoading, setVisualizationsLoading] =
    useState<boolean>(false);
  const [visualizationsLoaded, setVisualizationsLoaded] =
    useState<boolean>(false);
  const [isInitialLoad, setIsInitialLoad] = useState(true);

  useEffect(() => {
    if (isInitialLoad) {
      setIsInitialLoad(false);
      return;
    }

    const savedPatientDetails = localStorage.getItem("patientDetails");
    const savedSelectedModel = localStorage.getItem("selectedModel");
    const savedUploadedImage = localStorage.getItem("uploadedImage");
    const savedPrediction = localStorage.getItem("prediction");
    const savedConfidenceScore = localStorage.getItem("confidenceScore");
    const savedGradCamImage = localStorage.getItem("gradCamImage");
    const savedlimeImage = localStorage.getItem("limeImage");
    const savedDoctorNotes = localStorage.getItem("doctorNotes");

    if (savedPatientDetails) setPatientDetails(JSON.parse(savedPatientDetails));
    if (savedSelectedModel) setSelectedModel(savedSelectedModel);
    if (savedUploadedImage) setUploadedImage(savedUploadedImage);
    if (savedPrediction) setPrediction(savedPrediction);
    if (savedConfidenceScore) setConfidenceScore(savedConfidenceScore);
    if (savedGradCamImage) setGradCamImage(savedGradCamImage);
    if (savedlimeImage) setlimeImage(savedlimeImage);
    if (savedDoctorNotes) setDoctorNotes(savedDoctorNotes);
  }, [isInitialLoad]);

  useEffect(() => {
    if (!isInitialLoad) {
      localStorage.setItem("patientDetails", JSON.stringify(patientDetails));
      localStorage.setItem("selectedModel", selectedModel);
      localStorage.setItem(
        "visualizationsLoaded",
        JSON.stringify(visualizationsLoaded)
      );

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
    visualizationsLoaded,
  ]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setUploadedImage(URL.createObjectURL(file));
    }
  };

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
        <h3 className="text-lg font-medium text-gray-800 mb-3">
          Grad-CAM Guide
        </h3>
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
    setVisualizationsLoaded(false);

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("model", selectedModel);

    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/predict/",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );

      setPrediction(response.data.prediction);
      setConfidenceScore(response.data.confidence);
      loadVisualizations(formData);
    } catch (error) {
      console.error("Error analyzing image:", error);
      toast.error("Failed to analyze image.");
    } finally {
      setLoading(false);
    }
  };

  const loadVisualizations = async (formData: FormData) => {
    if (visualizationsLoaded) return;

    setVisualizationsLoading(true);

    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/visualize/",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );

      if (response.data.gradcam_image) {
        setGradCamImage(
          `data:image/jpeg;base64,${response.data.gradcam_image}`
        );
      }

      if (response.data.lime_image) {
        setlimeImage(`data:image/jpeg;base64,${response.data.lime_image}`);
      }

      setVisualizationsLoaded(true);
    } catch (error) {
      console.error("Error loading visualizations:", error);
      toast.error("Failed to load visualizations.");
    } finally {
      setVisualizationsLoading(false);
    }
  };

  const clearAllData = () => {
    localStorage.removeItem("patientDetails");
    localStorage.removeItem("selectedModel");
    localStorage.removeItem("uploadedImage");
    localStorage.removeItem("prediction");
    localStorage.removeItem("confidenceScore");
    localStorage.removeItem("gradCamImage");
    localStorage.removeItem("limeImage");
    localStorage.removeItem("doctorNotes");

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
    if (!patientDetails.name) {
      toast.error("Please enter patient name before generating the report.");
      return;
    }

    setReportDialogOpen(true);
    setIsReportGenerating(true);
    sendReportToDB();
    await new Promise((resolve) => setTimeout(resolve, 2000));
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
      formData.append("report_pdf", pdfBlob, "report.pdf"); 

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
    const pageHeight = doc.internal.pageSize.height;
    const margin = 20;
    const guideImage = "/image.png";

    const addHeader = () => {
      doc.setDrawColor(0, 102, 204);
      doc.setLineWidth(0.5);
      doc.line(margin, margin, pageWidth - margin, margin);

      doc.setFontSize(22);
      doc.setFont("helvetica", "bold");
      doc.setTextColor(0, 102, 204);
      doc.text("Osteo-Vision", margin, margin - 5);

      doc.setFontSize(10);
      doc.setFont("helvetica", "italic");
      doc.setTextColor(100, 100, 100);
      doc.text(
        "Advanced Osteoporosis Detection & Analysis",
        margin,
        margin + 5
      );

      doc.line(
        margin,
        pageHeight - margin,
        pageWidth - margin,
        pageHeight - margin
      );

      doc.setFontSize(8);
      doc.setFont("helvetica", "normal");
      doc.setTextColor(100, 100, 100);
      const pageInfo = `Page ${doc.internal.pages.length - 1}`;
      doc.text(pageInfo, pageWidth - margin, pageHeight - margin + 5, {
        align: "right",
      });
    };

    addHeader();

    doc.setFontSize(18);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(0, 0, 0);
    doc.text("KNEE OSTEOPOROSIS ANALYSIS REPORT", pageWidth / 2, 45, {
      align: "center",
    });

    doc.setDrawColor(0, 102, 204);
    doc.setLineWidth(0.3);
    doc.line(pageWidth / 2 - 40, 48, pageWidth / 2 + 40, 48);

    doc.setFillColor(240, 240, 240);
    doc.rect(margin, 60, pageWidth - margin * 2, 50, "F");

    doc.setFontSize(12);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(0, 0, 0);
    doc.text("PATIENT INFORMATION", margin + 5, 70);

    doc.setLineWidth(0.2);
    doc.setDrawColor(200, 200, 200);
    doc.line(margin + 5, 72, margin + 60, 72);

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

    doc.setFillColor(230, 240, 250);
    doc.rect(margin, 120, pageWidth - margin * 2, 40, "F");

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

    if (uploadedImage) {
      doc.addPage();
      addHeader();

      doc.setFontSize(14);
      doc.setFont("helvetica", "bold");
      doc.text("ORIGINAL X-RAY IMAGE", pageWidth / 2, 40, { align: "center" });

      doc.setFontSize(10);
      doc.setFont("helvetica", "italic");
      doc.text(
        "Patient knee radiograph submitted for analysis",
        pageWidth / 2,
        48,
        { align: "center" }
      );

      doc.setDrawColor(200, 200, 200);
      doc.setLineWidth(0.5);
      const imageX = (pageWidth - 120) / 2;
      doc.rect(imageX - 1, 55 - 1, 122, 122);
      doc.addImage(uploadedImage, "JPEG", imageX, 55, 120, 120);

      doc.setFontSize(9);
      doc.setFont("helvetica", "normal");
      doc.text("Fig. 1: Original knee radiograph", pageWidth / 2, 185, {
        align: "center",
      });
    }

    if (gradCamImage || limeImage) {
      doc.addPage();
      addHeader();

      doc.setFontSize(14);
      doc.setFont("helvetica", "bold");
      doc.text("AI ANALYSIS VISUALIZATIONS", pageWidth / 2, 40, {
        align: "center",
      });

      doc.setFontSize(9);
      doc.setFont("helvetica", "italic");
      doc.text(
        "The following images show areas analyzed by the AI model to reach its diagnosis",
        pageWidth / 2,
        48,
        { align: "center" }
      );

      if (gradCamImage && limeImage) {
        const imgWidth = 80;
        const leftImgX = margin + 10;
        const rightImgX = pageWidth - margin - imgWidth - 10;
        const imgY = 60;

        doc.setDrawColor(200, 200, 200);
        doc.setLineWidth(0.5);
        doc.rect(leftImgX - 1, imgY - 1, imgWidth + 2, imgWidth + 2);
        doc.addImage(gradCamImage, "JPEG", leftImgX, imgY, imgWidth, imgWidth);

        doc.rect(rightImgX - 1, imgY - 1, imgWidth + 2, imgWidth + 2);
        doc.addImage(limeImage, "JPEG", rightImgX, imgY, imgWidth, imgWidth);

        doc.setFontSize(9);
        doc.setFont("helvetica", "normal");
        doc.text(
          "Fig. 2: Grad-CAM Heatmap",
          leftImgX + imgWidth / 2,
          imgY + imgWidth + 10,
          { align: "center" }
        );
        doc.text(
          "Fig. 3: LIME Explanations",
          rightImgX + imgWidth / 2,
          imgY + imgWidth + 10,
          { align: "center" }
        );

        const guideWidth = 50;
        const guideX = (pageWidth - guideWidth) / 2;
        doc.addImage(
          guideImage,
          "PNG",
          guideX,
          imgY + imgWidth + 20,
          guideWidth,
          guideWidth
        );

        doc.setFontSize(9);
        doc.setFont("helvetica", "italic");
        doc.text("Interpretation Guide", pageWidth / 2, imgY + imgWidth + 80, {
          align: "center",
        });

        doc.setFontSize(9);
        doc.setFont("helvetica", "normal");
        const explanationText =
          "Grad-CAM (Gradient-weighted Class Activation Mapping) highlights regions that influenced the AI's decision most significantly. " +
          "Red areas indicate regions of highest importance for the diagnosis. " +
          "LIME (Local Interpretable Model-agnostic Explanations) identifies specific features that contributed to the prediction. " +
          "These visualizations help clinicians understand how the AI reached its conclusion.";

        doc.text(explanationText, margin, imgY + imgWidth + 90, {
          align: "left",
          maxWidth: pageWidth - margin * 2,
        });
      } else if (gradCamImage) {
        const imgWidth = 110;
        const imgX = (pageWidth - imgWidth) / 2;
        doc.addImage(gradCamImage, "JPEG", imgX, 60, imgWidth, imgWidth);
        doc.text("Fig. 2: Grad-CAM Heatmap", pageWidth / 2, 180, {
          align: "center",
        });
        doc.addImage(guideImage, "PNG", (pageWidth - 40) / 2, 190, 40, 40);
      } else if (limeImage) {
        const imgWidth = 110;
        const imgX = (pageWidth - imgWidth) / 2;
        doc.addImage(limeImage, "JPEG", imgX, 60, imgWidth, imgWidth);
        doc.text("Fig. 2: LIME Explanations", pageWidth / 2, 180, {
          align: "center",
        });
      }
    }

    doc.addPage();
    addHeader();

    doc.setFontSize(14);
    doc.setFont("helvetica", "bold");
    doc.text("CLINICAL ASSESSMENT", pageWidth / 2, 40, { align: "center" });

    doc.setDrawColor(0, 102, 204);
    doc.setLineWidth(0.3);
    doc.line(pageWidth / 2 - 30, 43, pageWidth / 2 + 30, 43);

    doc.setDrawColor(220, 220, 220);
    doc.setLineWidth(0.5);
    doc.rect(margin, 50, pageWidth - margin * 2, pageHeight - 100);

    doc.setFontSize(11);
    doc.setFont("helvetica", "bold");
    doc.text("Doctor's Notes:", margin + 5, 60);

    doc.setFontSize(10);
    doc.setFont("helvetica", "normal");
    doc.text(doctorNotes, margin + 5, 70, {
      maxWidth: pageWidth - margin * 2 - 10,
    });

    doc.text(
      "Physician's Signature: _________________________",
      margin + 5,
      pageHeight - 40
    );
    doc.text("Date: _________________", pageWidth - 70, pageHeight - 40);

    return doc;
  };

  const confirmReportDownload = async () => {
    const doc = await formatReport();
    const sanitizedFileName = patientDetails.name
      .replace(/[^a-z0-9]/gi, "_")
      .toLowerCase();
    doc.save(`Osteo_Vision_Report_${sanitizedFileName}.pdf`);
    setReportDialogOpen(false);
  };

  const cancelReportDownload = () => {
    setReportDialogOpen(false);
  };

  return (
    <div className="pt-[64px]">
    <div className="relative min-h-screen bg-gradient-to-r from-blue-500 to-teal-400 flex items-center justify-center p-6">
      
        {/* Background image */}
        <div
          className="absolute inset-0 bg-cover bg-center opacity-20 z-0"
          style={{ backgroundImage: 'url("/background.jpg")' }}
        ></div>

<div className="relative z-10 bg-white rounded-2xl shadow-2xl p-10 w-full max-w-6xl">
          {/* Header */}
          <div className="flex justify-between items-center mb-6">
            <div>
              <h1 className="text-2xl md:text-3xl font-bold text-gray-800">
                Osteoporosis Report Generator
              </h1>
              <p className="text-gray-600">
                AI-powered knee osteoporosis analysis
              </p>
            </div>
            <button
              onClick={clearAllData}
              className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-red-600 hover:text-red-700 bg-red-50 hover:bg-red-100 rounded-lg transition-colors"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-5 w-5"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  fillRule="evenodd"
                  d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z"
                  clipRule="evenodd"
                />
              </svg>
              Clear All
            </button>
          </div>

          {/* Main Card */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
            <div className="p-6 md:p-8 space-y-6">
              {/* Patient Information */}
              <div className="space-y-4">
                <h2 className="text-xl font-semibold text-gray-800">
                  Patient Information
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Full Name
                    </label>
                    <input
                      type="text"
                      placeholder="Patient Name"
                      value={patientDetails.name}
                      onChange={(e) =>
                        setPatientDetails({
                          ...patientDetails,
                          name: e.target.value,
                        })
                      }
                      className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Patient ID
                    </label>
                    <input
                      type="text"
                      placeholder="Patient ID"
                      value={patientDetails.id}
                      onChange={(e) =>
                        setPatientDetails({
                          ...patientDetails,
                          id: e.target.value,
                        })
                      }
                      className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Age
                    </label>
                    <input
                      type="number"
                      placeholder="Age"
                      value={patientDetails.age}
                      onChange={(e) =>
                        setPatientDetails({
                          ...patientDetails,
                          age: e.target.value,
                        })
                      }
                      className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Gender
                    </label>
                    <select
                      value={patientDetails.gender}
                      onChange={(e) =>
                        setPatientDetails({
                          ...patientDetails,
                          gender: e.target.value,
                        })
                      }
                      className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                    >
                      <option value="">Select Gender</option>
                      <option value="Male">Male</option>
                      <option value="Female">Female</option>
                      <option value="Other">Other</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Model Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  AI Model
                </label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                >
                  <option value="binary_vgg19">Binary - VGG19</option>
                  <option value="efficientnet">Binary - EfficientNet</option>
                  <option value="ensemble">Binary - Ensemble</option>
                  <option value="multiclass_vgg19">Multiclass - VGG19</option>
                </select>
              </div>

              {/* Image Upload */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Knee X-ray Image
                </label>
                <div className="flex flex-col sm:flex-row gap-4">
                  <label className="flex-1 cursor-pointer">
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleFileChange}
                      className="hidden"
                    />
                    <div className="w-full h-full p-8 border-2 border-dashed border-gray-300 rounded-lg hover:border-blue-400 transition-colors flex flex-col items-center justify-center">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="h-10 w-10 text-gray-400 mb-2"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={1.5}
                          d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                        />
                      </svg>
                      <span className="text-gray-600">
                        Click to upload or drag and drop
                      </span>
                      <span className="text-sm text-gray-500 mt-1">
                        JPG, PNG (Max 5MB)
                      </span>
                    </div>
                  </label>
                  {uploadedImage && (
                    <div className="flex-1 flex items-center justify-center">
                      <div className="relative w-full max-w-xs aspect-square">
                        <img
                          src={uploadedImage}
                          alt="Uploaded X-ray"
                          className={`w-full h-full object-contain rounded-lg border border-gray-200 transition-all duration-300 ${
                            loading ? "opacity-70" : "opacity-100"
                          }`}
                        />
                        {loading && (
                          <div className="absolute inset-0 flex items-center justify-center">
                            <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Analyze Button */}
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
                    <svg
                      className="animate-spin h-5 w-5 text-white"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      ></circle>
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      ></path>
                    </svg>
                    Analyzing...
                  </div>
                ) : (
                  "Analyze Image"
                )}
              </button>

              {/* Results Section */}
              {(prediction || confidenceScore) && (
                <div className="space-y-4">
                  <h2 className="text-xl font-semibold text-gray-800">
                    Analysis Results
                  </h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-blue-50 p-4 rounded-lg">
                      <h3 className="text-sm font-medium text-blue-800 mb-1">
                        Prediction
                      </h3>
                      <p className="text-2xl font-bold text-blue-900">
                        {prediction}
                      </p>
                    </div>
                    <div className="bg-blue-50 p-4 rounded-lg">
                      <h3 className="text-sm font-medium text-blue-800 mb-1">
                        Confidence Score
                      </h3>
                      <p className="text-2xl font-bold text-blue-900">
                        {confidenceScore}
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Visualizations */}
              {visualizationsLoading && (
                <div className="space-y-4">
                  <h2 className="text-xl font-semibold text-gray-800">
                    Generating Visualizations
                  </h2>
                  <div className="space-y-2">
                    <div className="h-2.5 w-full bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-blue-600 rounded-full animate-pulse"
                        style={{ width: "70%" }}
                      ></div>
                    </div>
                    <p className="text-sm text-gray-500">
                      Generating visual explanations...
                    </p>
                  </div>
                </div>
              )}

              {(gradCamImage || limeImage) && (
                <div className="space-y-4">
                  <h2 className="text-xl font-semibold text-gray-800">
                    AI Explanations
                  </h2>
                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {gradCamImage && (
                      <div className="space-y-2">
                        <h3 className="text-lg font-medium text-gray-700">
                          Grad-CAM Heatmap
                        </h3>
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
                      <h3 className="text-lg font-medium text-gray-700">
                        Interpretation Guide
                      </h3>
                      <ColorLegend />
                    </div>
                    {limeImage && (
                      <div className="space-y-2">
                        <h3 className="text-lg font-medium text-gray-700">
                          LIME Visualization
                        </h3>
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
                </div>
              )}

              {/* Doctor's Notes */}
              <div className="space-y-2">
                <label className="block text-sm font-medium text-gray-700">
                  Clinical Notes
                </label>
                <textarea
                  placeholder="Enter your clinical observations and recommendations..."
                  value={doctorNotes}
                  onChange={(e) => setDoctorNotes(e.target.value)}
                  rows={4}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                ></textarea>
              </div>

              {/* Generate Report Button */}
              <button
                onClick={generateReport}
                disabled={!patientDetails.name || !prediction}
                className={`w-full py-3 px-6 rounded-lg font-medium text-white transition-colors ${
                  !patientDetails.name || !prediction
                    ? "bg-gray-400 cursor-not-allowed"
                    : "bg-green-600 hover:bg-green-700"
                }`}
              >
                Generate PDF Report
              </button>
            </div>
          </div>
        </div>

        {/* Report Generation Dialog */}
        <ReportGenerationDialog
          open={reportDialogOpen}
          onOpenChange={setReportDialogOpen}
          isGenerating={isReportGenerating}
          onConfirm={confirmReportDownload}
          onCancel={cancelReportDownload}
        />

        <ToastContainer />
      </div>
    </div>
  );
}
