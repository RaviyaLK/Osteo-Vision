/* eslint-disable @next/next/no-img-element */
"use client";
import { useState, useEffect } from "react";
import { FiDownload } from "react-icons/fi";
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
    <div className="relative min-h-screen bg-linear-to-r from-blue-500 to-teal-400 p-8 flex items-center justify-center">
      <div
        className="absolute inset-0 bg-cover bg-center opacity-40"
        style={{ backgroundImage: 'url("/background.jpg")' }}
      ></div>

      <div className="relative z-10 container mx-auto p-8 bg-white shadow-xl rounded-lg max-w-4xl max-h-[1000px] ">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-3xl font-extrabold text-gray-800">
            Medical History
          </h1>
         <AlertDialog>
  <AlertDialogTrigger asChild>
    <button
      className="bg-red-500 text-white py-2 px-4 rounded-lg hover:bg-red-600 transition-all duration-200"
    >
      Clear All Reports
    </button>
  </AlertDialogTrigger>
  <AlertDialogContent>
    <AlertDialogHeader>
      <AlertDialogTitle>Are you absolutely sure?</AlertDialogTitle>
      <AlertDialogDescription>
        This action cannot be undone. This will permanently delete all medical reports from the database.
      </AlertDialogDescription>
    </AlertDialogHeader>
    <AlertDialogFooter>
      <AlertDialogCancel>Cancel</AlertDialogCancel>
      <AlertDialogAction
        className="bg-red-600 hover:bg-red-700"
        onClick={clearDatabase}
      >
        Continue
      </AlertDialogAction>
    </AlertDialogFooter>
  </AlertDialogContent>
</AlertDialog>
        </div>

        <p className="mb-6 text-gray-600">
          Your medical history and previous diagnoses are shown here.
        </p>

        <div className="max-h-[800px] overflow-y-auto border-t border-b border-gray-200">
          {loading ? (
            <p className="text-gray-500">Loading reports...</p>
          ) : reports.length === 0 ? (
            <div className="flex justify-center items-center py-20">
              <div className="p-10 rounded-lg text-center w-full max-w-md">
                <div className="mb-5">
                  <img
                    src="/not_found.png"
                    alt="No reports"
                    className="mx-auto w-24 h-24 object-contain"
                  />
                </div>
                <p className="text-gray-600 text-xl mb-4">No reports found.</p>
                <p className="text-md text-gray-500">
                  Looks like you haven&apos;t uploaded any reports yet.
                </p>
              </div>
            </div>
          ) : (
            <ul className="space-y-6 p-4">
              {reports.map((report) => (
                <li
                  key={report._id}
                  className="bg-gray-50 p-6 shadow-md rounded-lg flex items-center justify-between transition-all hover:shadow-xl"
                >
                  <div>
                    <h2 className="text-lg font-semibold text-gray-800">
                      {report.patient_name}
                    </h2>
                    <p className="text-sm text-gray-600">
                      <strong>Model Used:</strong> {report.model_used}
                    </p>
                    <p className="text-sm text-gray-600">
                      <strong>Prediction:</strong> {report.prediction} (
                      {(report.confidence_score * 100).toFixed(2)}%)
                    </p>
                    <p className="text-sm text-gray-600">
                      <strong>Generated on:</strong>{" "}
                      {new Date(report.generated_on).toLocaleString()}
                    </p>
                  </div>

                  <button
                    className={`bg-blue-500 text-white p-3 rounded-full hover:bg-blue-600 transition-all duration-300 relative ${
                      loadingReportId === report._id ? "cursor-wait" : ""
                    }`}
                    onClick={() =>
                      downloadReport(report._id, report.patient_name)
                    }
                    disabled={loadingReportId === report._id}
                  >
                    {loadingReportId === report._id ? (
                      <Spinner
                        animation="border"
                        size="sm"
                        className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2"
                      />
                    ) : (
                      <FiDownload size={20} />
                    )}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
      <ToastContainer />
    </div>
  );
}
