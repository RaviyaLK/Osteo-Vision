/* eslint-disable react/no-unescaped-entities */
"use client";
import { FaUpload, FaCogs, FaChartBar, FaFileDownload, FaHistory, FaUserShield } from "react-icons/fa";

export default function DocumentationPage() {
  return (
    <div className="pt-[60px]">
 <div className="relative min-h-screen bg-gradient-to-r from-blue-500 to-teal-400 flex items-center justify-center p-6">
      {/* Background image */}
      <div
        className="absolute inset-0 bg-cover bg-center opacity-20 z-0"
        style={{ backgroundImage: 'url("/background.jpg")' }}
      />

      <div className="max-w-6xl mx-auto relative z-10">
        {/* Main Card */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
          <div className="p-6 md:p-8 space-y-8">
            {/* Header */}
            <div className="text-center">
              <h1 className="text-3xl md:text-4xl font-bold text-gray-800 mb-4">
                OsteoVision System Guide
              </h1>
              <p className="text-lg text-gray-600 max-w-3xl mx-auto">
                Comprehensive guide to using Osteo-Vision knee osteoporosis detection platform.    
                Learn how to upload scans, interpret results, and manage your medical history.
              </p>
            </div>

            {/* Divider */}
            <div className="border-t border-gray-200"></div>

            {/* Steps Section */}
            <div className="space-y-10">
              {/* Step 1 */}
              <div className="flex flex-col md:flex-row gap-6 p-4 md:p-6 bg-blue-50 rounded-lg">
                <div className="flex items-center justify-center w-16 h-16 bg-blue-100 text-blue-600 rounded-full shrink-0">
                  <FaUpload className="text-2xl" />
                </div>
                <div>
                  <h2 className="text-2xl font-semibold text-gray-800 mb-3">
                    Step 1: Upload X-ray Image
                  </h2>
                  <div className="space-y-3 text-gray-700">
                    <p>
                      Navigate to the <strong>Home</strong> page and click the upload area or drag-and-drop your knee X-ray image. 
                      The system supports standard JPEG, and PNG formats.
                    </p>
                    <p className="bg-blue-100 p-3 rounded-md border border-blue-200">
                      <strong>Pro Tip:</strong> For best results, use high-quality images with clear bone structure visibility. 
                      Ensure the knee joint is centered in the image.
                    </p>
                  </div>
                </div>
              </div>

              {/* Step 2 */}
              <div className="flex flex-col md:flex-row gap-6 p-4 md:p-6 bg-purple-50 rounded-lg">
                <div className="flex items-center justify-center w-16 h-16 bg-purple-100 text-purple-600 rounded-full shrink-0">
                  <FaCogs className="text-2xl" />
                </div>
                <div>
                  <h2 className="text-2xl font-semibold text-gray-800 mb-3">
                    Step 2: Select Analysis Model
                  </h2>
                  <div className="space-y-3 text-gray-700">
                    <p>
                      Choose between our specialized AI models:
                    </p>
                    <ul className="list-disc pl-5 space-y-2">
                      <li>
                        <strong>Binary Classification (VGG19, EfficientNet, Ensemble):</strong> Determines if the knee shows signs of osteoporosis (Healthy vs Osteoporotic)
                      </li>
                      <li>
                        <strong>Multiclass Classification:</strong> Differentiates between Healthy, Osteopenia (early bone loss), and Osteoporosis
                      </li>
                      
                    </ul>
                    <p>
                      Select the optimum model for your case.
                    </p>
                  </div>
                </div>
              </div>

              {/* Step 3 */}
              <div className="flex flex-col md:flex-row gap-6 p-4 md:p-6 bg-green-50 rounded-lg">
                <div className="flex items-center justify-center w-16 h-16 bg-green-100 text-green-600 rounded-full shrink-0">
                  <FaChartBar className="text-2xl" />
                </div>
                <div>
                  <h2 className="text-2xl font-semibold text-gray-800 mb-3">
                    Step 3: Interpret Results
                  </h2>
                  <div className="space-y-3 text-gray-700">
                    <p>
                      After processing (typically 15-30 seconds), you'll see:
                    </p>
                    <ul className="list-disc pl-5 space-y-2">
                      <li>
                        <strong>Diagnosis:</strong> Clear classification of bone health status
                      </li>
                      <li>
                        <strong>Confidence Score:</strong> Percentage indicating the model's certainty
                      </li>
                      <li>
                        <strong>Visual Explanations:</strong> Heatmaps showing areas that influenced the diagnosis
                      </li>
                    </ul>
                    <div className="bg-green-100 p-3 rounded-md border border-green-200">
                      <strong>Understanding Heatmaps:</strong> The color gradient (blue to red) indicates regions of interest, 
                      with red areas showing the strongest indicators of bone density changes.
                    </div>
                  </div>
                </div>
              </div>

              {/* Step 4 */}
              <div className="flex flex-col md:flex-row gap-6 p-4 md:p-6 bg-yellow-50 rounded-lg">
                <div className="flex items-center justify-center w-16 h-16 bg-yellow-100 text-yellow-600 rounded-full shrink-0">
                  <FaFileDownload className="text-2xl" />
                </div>
                <div>
                  <h2 className="text-2xl font-semibold text-gray-800 mb-3">
                    Step 4: Generate & Download Report
                  </h2>
                  <div className="space-y-3 text-gray-700">
                    <p>
                      Click the "Generate Report" button to create a comprehensive PDF containing:
                    </p>
                    <ul className="list-disc pl-5 space-y-2">
                      <li>Patient information (name, ID, age, gender)</li>
                      <li>Original X-ray image with annotations</li>
                      <li>Detailed analysis results</li>
                      <li>AI explanation visualizations</li>
                      <li>Clinical notes section</li>
                    </ul>
                    <p>
                      Reports are automatically saved to your medical history for future reference.
                    </p>
                  </div>
                </div>
              </div>

              {/* Step 5 */}
              <div className="flex flex-col md:flex-row gap-6 p-4 md:p-6 bg-indigo-50 rounded-lg">
                <div className="flex items-center justify-center w-16 h-16 bg-indigo-100 text-indigo-600 rounded-full shrink-0">
                  <FaHistory className="text-2xl" />
                </div>
                <div>
                  <h2 className="text-2xl font-semibold text-gray-800 mb-3">
                    Step 5: Manage Medical History
                  </h2>
                  <div className="space-y-3 text-gray-700">
                    <p>
                      The <strong>History</strong> page provides access to all previous reports with:
                    </p>
                    <ul className="list-disc pl-5 space-y-2">
                      <li>Chronological listing of all analyses</li>
                      <li>Quick-view of diagnosis and confidence scores</li>
                      <li>Download capability for any previous report</li>
                   
                    </ul>
                    <div className="bg-indigo-100 p-3 rounded-md border border-indigo-200">
                      <strong>Data Security:</strong> All medical data is encrypted and stored securely in compliance with HIPAA regulations.
                    </div>
                  </div>
                </div>
              </div>

              {/* Step 6 */}
              <div className="flex flex-col md:flex-row gap-6 p-4 md:p-6 bg-red-50 rounded-lg">
                <div className="flex items-center justify-center w-16 h-16 bg-red-100 text-red-600 rounded-full shrink-0">
                  <FaUserShield className="text-2xl" />
                </div>
                <div>
                  <h2 className="text-2xl font-semibold text-gray-800 mb-3">
                    Step 6: Administrative Functions
                  </h2>
                  <div className="space-y-3 text-gray-700">
                    <p>
                      For authorized administrators only:
                    </p>
                    <ul className="list-disc pl-5 space-y-2">
                      <li>
                        <strong>Database Management:</strong> Clear all patient records when needed
                      </li>
                    
                    </ul>
                    <div className="bg-red-100 p-3 rounded-md border border-red-200">
                      <strong>Warning:</strong> Administrative functions affect all user data. 
                      Always confirm actions and maintain proper backups.
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* System Overview */}
            <div className="mt-12 p-6 bg-gray-50 rounded-lg border border-gray-200">
              <h2 className="text-2xl font-semibold text-gray-800 mb-4">System Overview</h2>
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-xl font-medium text-gray-700 mb-3">Technical Specifications</h3>
                  <ul className="space-y-2 text-gray-600">
                    <li>• AI Models: VGG19, EfficientNet, Ensemble, Multi-class</li>
                    <li>• Processing Time: 15-30 seconds per image</li>
                    <li>• Accuracy: 84-98.52% across test datasets</li>
                
                  </ul>
                </div>
                <div>
                  <h3 className="text-xl font-medium text-gray-700 mb-3">Clinical Best Practices</h3>
                  <ul className="space-y-2 text-gray-600">
                    <li>• Always correlate AI results with clinical findings</li>
                    <li>• Use the system as a decision support tool</li>
                  
                   
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    </div>
  );
}