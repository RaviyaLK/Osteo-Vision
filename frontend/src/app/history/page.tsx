/* eslint-disable @next/next/no-img-element */
"use client";
import { useState, useEffect } from "react";
import { FiDownload, FiTrash2 } from "react-icons/fi";
import { toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import { ToastContainer } from "react-toastify";
import { Spinner } from "react-bootstrap";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"

export default function HistoryPage() {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [reports, setReports] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingReportId, setLoadingReportId] = useState<string | null>(null);

  useEffect(() => {
    fetch("http://127.0.0.1:8000/get-reports/")
      .then((response) => response.json())
      .then((data) => {
        setReports(data);
        setLoading(false);
      })
      .catch((error) => {
        console.error("Error fetching reports:", error);
        toast.error("Failed to load medical history");
        setLoading(false);
      });
  }, []);

  const downloadReport = (reportId: string, patientName: string) => {
    setLoadingReportId(reportId);
    const url = `http://127.0.0.1:8000/download-report/${reportId}`;
    fetch(url)
      .then((response) => {
        if (response.ok) {
          return response.blob();
        } else {
          throw new Error("Failed to download report");
        }
      })
      .then((blob) => {
        const downloadUrl = window.URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = downloadUrl;
        link.download = `${patientName}_report.pdf`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        setLoadingReportId(null);
        toast.success("Report downloaded successfully!");
      })
      .catch((error) => {
        console.error("Download error:", error);
        toast.error("Error downloading report!");
        setLoadingReportId(null);
      });
  };

  const clearDatabase = () => {
    if (reports.length === 0) {
      toast.info("Medical history is already empty.");
      return;
    }

    fetch("http://127.0.0.1:8000/clear-reports/", {
      method: "DELETE",
    })
      .then((response) => {
        if (response.ok) {
          setReports([]);
          toast.success("History cleared successfully!");
        } else {
          toast.error("Failed to clear History");
        }
      })
      .catch((error) => {
        console.error("Error clearing the database:", error);
        toast.error("Error clearing the History.");
      });
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
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
              <div>
                <h1 className="text-2xl md:text-3xl font-bold text-gray-800">Medical History</h1>
                <p className="text-gray-600 mt-1">Your previous diagnoses and reports</p>
              </div>
              
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <button className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-red-600 hover:text-red-700 bg-red-50 hover:bg-red-100 rounded-lg transition-colors">
                    <FiTrash2 className="h-4 w-4" />
                    Clear All Reports
                  </button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Confirm Clear History</AlertDialogTitle>
                    <AlertDialogDescription>
                      This will permanently delete all medical reports from the database. This action cannot be undone.
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <AlertDialogAction
                      className="bg-red-600 hover:bg-red-700"
                      onClick={clearDatabase}
                    >
                      Clear History
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            </div>

            {/* Reports List */}
            <div className="border border-gray-200 rounded-lg overflow-hidden">
              {loading ? (
                <div className="flex justify-center items-center py-20">
                  <div className="flex flex-col items-center gap-4">
                    <Spinner animation="border" variant="primary" />
                    <p className="text-gray-600">Loading medical history...</p>
                  </div>
                </div>
              ) : reports.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-16 px-4 text-center">
                  <img
                    src="/not_found.png"
                    alt="No reports"
                    className="w-32 h-32 object-contain mb-4 opacity-70"
                  />
                  <h3 className="text-lg font-medium text-gray-800 mb-2">No Reports Found</h3>
                  <p className="text-gray-600 max-w-md">
                    Your medical history will appear here after you generate and save reports.
                  </p>
                </div>
              ) : (
                <ul className="divide-y divide-gray-200 max-h-[600px] overflow-y-auto">
                  {reports.map((report) => (
                    <li key={report._id} className="p-4 hover:bg-gray-50 transition-colors">
                      <div className="flex items-center justify-between gap-4">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-3 mb-2">
                            <h3 className="text-lg font-semibold text-gray-800 truncate">
                              {report.patient_name}
                            </h3>
                            <span className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded-full">
                              {report.model_used.replace(/_/g, ' ')}
                            </span>
                          </div>
                          
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm text-gray-600">
                            <div>
                              <span className="font-medium">Prediction:</span> {report.prediction}
                            </div>
                            <div>
                              <span className="font-medium">Confidence:</span> {(report.confidence_score * 100).toFixed(2)}%
                            </div>
                            <div className="md:col-span-2">
                              <span className="font-medium">Generated:</span> {new Date(report.generated_on).toLocaleString()}
                            </div>
                          </div>
                        </div>

                        <button
                          onClick={() => downloadReport(report._id, report.patient_name)}
                          disabled={loadingReportId === report._id}
                          className={`p-3 rounded-full text-white transition-colors ${
                            loadingReportId === report._id 
                              ? "bg-blue-400 cursor-wait" 
                              : "bg-blue-600 hover:bg-blue-700"
                          }`}
                        >
                          {loadingReportId === report._id ? (
                            <Spinner animation="border" size="sm" className="text-white" />
                          ) : (
                            <FiDownload className="h-5 w-5" />
                          )}
                        </button>
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Toast Notifications */}
      <ToastContainer 
    
      />
    </div>
    </div>
  );
}