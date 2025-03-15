"use client";

import { useState } from "react";
import axios from "axios";
import { jsPDF } from "jspdf";

export default function ReportPage() {
  const [patientDetails, setPatientDetails] = useState({
    name: "",
    id: "",
    age: "",
    gender: "",
  });
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState("");
  const [confidenceScore, setConfidenceScore] = useState("");
  const [gradCamImage, setGradCamImage] = useState("");
  const [doctorNotes, setDoctorNotes] = useState("");
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert("Please upload an X-ray image.");
      return;
    }
    setLoading(true);
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await axios.post("http://127.0.0.1:8001/upload/", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setPrediction(response.data.prediction);
      setConfidenceScore(response.data.confidence);
      setGradCamImage(`data:image/jpeg;base64,${response.data.gradcam_image}`);
    } catch (error) {
      console.error("Error uploading image:", error);
      alert("Failed to analyze image.");
    } finally {
      setLoading(false);
    }
  };

  const generateReport = () => {
    const doc = new jsPDF();
    const dateTime = new Date().toLocaleString();

    doc.setFontSize(16);
    doc.text("Knee Osteoporosis Analysis Report", 20, 20);

    doc.setFontSize(10);
    doc.text(`Generated on: ${dateTime}`, 20, 30);
    doc.text(`Patient Name: ${patientDetails.name}`, 20, 40);
    doc.text(`Patient ID: ${patientDetails.id}`, 20, 50);
    doc.text(`Age: ${patientDetails.age}`, 20, 60);
    doc.text(`Gender: ${patientDetails.gender}`, 20, 70);

    doc.text(`Prediction: ${prediction}`, 20, 90);
    doc.text(`Confidence Score: ${confidenceScore}`, 20, 100);

    doc.text("Doctor's Notes:", 20, 120);
    doc.text(doctorNotes, 20, 130, { maxWidth: 170 });

    doc.save("Knee_Osteoporosis_Report.pdf");
  };

  return (
    <div className="relative min-h-screen bg-gray-100 p-6 flex items-center justify-center">
      <div
        className="absolute inset-0 bg-cover bg-center opacity-30"
        style={{ backgroundImage: 'url("/background.jpg")' }}
      ></div>

      <div className="relative z-10 flex flex-col items-center justify-center w-full max-w-3xl">
        <div className="bg-white shadow rounded-md p-10 w-full">
          <h1 className="text-2xl font-bold mb-4 text-center">
            Generate Medical Report
          </h1>

          {/* Patient Details */}
          <input
            type="text"
            placeholder="Patient Name"
            value={patientDetails.name}
            onChange={(e) =>
              setPatientDetails({ ...patientDetails, name: e.target.value })
            }
            className="w-full p-2.5 mb-3 border rounded-md text-base"
          />
          <input
            type="text"
            placeholder="Patient ID"
            value={patientDetails.id}
            onChange={(e) =>
              setPatientDetails({ ...patientDetails, id: e.target.value })
            }
            className="w-full p-2.5 mb-3 border rounded-md text-base"
          />
          <input
            type="number"
            placeholder="Age"
            value={patientDetails.age}
            onChange={(e) =>
              setPatientDetails({ ...patientDetails, age: e.target.value })
            }
            className="w-full p-2.5 mb-3 border rounded-md text-base"
          />
          <select
            value={patientDetails.gender}
            onChange={(e) =>
              setPatientDetails({ ...patientDetails, gender: e.target.value })
            }
            className="w-full p-2.5 mb-4 border rounded-md text-base"
          >
            <option value="">Select Gender</option>
            <option value="Male">Male</option>
            <option value="Female">Female</option>
            <option value="Other">Other</option>
          </select>

          {/* File Upload */}
          <div className="mb-4">
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              id="file-upload"
              className="hidden"
            />
            <label
              htmlFor="file-upload"
              className="cursor-pointer inline-block w-full text-center py-2.5 px-4 bg-blue-600 hover:bg-blue-800 text-white font-medium rounded-md text-base"
            >
              Select X-ray Image
            </label>
            {selectedFile && (
              <p className="mt-2 text-sm text-gray-700 text-center">
                {selectedFile.name}
              </p>
            )}
          </div>

          <button
            onClick={handleUpload}
            disabled={loading}
            className="w-full text-center py-2.5 px-4 bg-blue-600 hover:bg-blue-800 text-white font-medium rounded-md text-base"
          >
            {loading ? "Processing..." : "Upload & Analyze"}
          </button>

          {/* Prediction */}
          {prediction && (
            <div className="mt-4 text-center">
              <h2 className="text-xl font-semibold">Prediction:</h2>
              <p className="text-blue-600 text-lg">{prediction}</p>
            </div>
          )}

          {/* Confidence Score */}
          {confidenceScore && (
            <div className="mt-3 text-center">
              <h2 className="text-xl font-semibold">Confidence Score:</h2>
              <p className="text-gray-700 text-lg">{confidenceScore}</p>
            </div>
          )}

          {/* Grad-CAM Image */}
          {gradCamImage && (
            <div className="mt-5">
              <h3 className="text-lg font-medium text-center">Grad-CAM Heatmap</h3>
              <img
                src={gradCamImage}
                alt="Grad-CAM Overlay"
                className="mt-3 rounded-md shadow mx-auto max-w-full"
              />
            </div>
          )}

          {/* Doctor Notes */}
          <textarea
            placeholder="Doctor's Notes"
            value={doctorNotes}
            onChange={(e) => setDoctorNotes(e.target.value)}
            className="w-full p-3 mt-4 border rounded-md text-base"
          ></textarea>

          {/* Generate Report */}
          <button
            onClick={generateReport}
            className="w-full py-2.5 mt-5 bg-green-600 hover:bg-green-800 text-white rounded-md text-base"
          >
            Generate Report
          </button>
        </div>
      </div>
    </div>
  );
}
