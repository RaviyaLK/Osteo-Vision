"use client";
import { FaArrowCircleRight } from "react-icons/fa";

export default function DocumentationPage() {
  return (
    <div className="relative min-h-screen bg-linear-to-r from-blue-500 to-teal-400 p-8 flex items-center justify-center">
      <div
        className="absolute inset-0 bg-cover bg-center opacity-40"
        style={{ backgroundImage: 'url("/background.jpg")' }}
      ></div>
      <div className="relative z-10 container mx-auto p-16 bg-white shadow-xl rounded-md max-w-6xl">
        <h1 className="text-4xl font-extrabold text-gray-800 mb-7 text-center  ">
          Step-by-Step Guide
        </h1>
        <p className="text-lg text-gray-600 mb-6">
          Welcome to the knee osteoporosis detection platform! Follow these
          steps to make the most of the website.
        </p>

        {/* Steps */}
        <div className="space-y-8">
          <div className="flex items-start space-x-4">
            <div className="flex items-center justify-center w-12 h-12  text-black rounded-xl">
              <FaArrowCircleRight size={24} />
            </div>
            <div>
              <h3 className="text-2xl font-semibold text-gray-800 mb-2">
                Step 1: Upload X-ray Image
              </h3>
              <p className="text-gray-600">
                Click the Upload button on the homepage to upload an X-ray image
                of the knee. Ensure that the image is clear and of high quality
                for accurate results.
              </p>
            </div>
          </div>

          <div className="flex items-start space-x-4">
            <div className="flex items-center justify-center w-12 h-12  text-black rounded-xl">
              <FaArrowCircleRight size={24} />
            </div>
            <div>
              <h3 className="text-2xl font-semibold text-gray-800 mb-2">
                Step 2: Select Model
              </h3>
              <p className="text-gray-600">
                After uploading the image, choose the desired model (Binary or
                Multiclass) from the dropdown. The binary model classifies the
                knee as healthy or osteoporotic, while the multiclass model also
                includes Osteopenia as a classification.
              </p>
            </div>
          </div>

          <div className="flex items-start space-x-4">
            <div className="flex items-center justify-center w-12 h-12  text-black rounded-xl">
              <FaArrowCircleRight size={24} />
            </div>
            <div>
              <h3 className="text-2xl font-semibold text-gray-800 mb-2">
                Step 3: View Results
              </h3>
              <p className="text-gray-600">
                After the model processes the image, you will see the prediction
                result along with the confidence score. If desired, you can also
                view the Grad-CAM overlay highlighting the regions that
                contributed to the model&apos;s decision.
              </p>
            </div>
          </div>

          <div className="flex items-start space-x-4">
            <div className="fflex items-center justify-center w-12 h-12  text-black rounded-xl">
              <FaArrowCircleRight size={24} />
            </div>
            <div>
              <h3 className="text-2xl font-semibold text-gray-800 mb-2">
                Step 4: Download Report
              </h3>
              <p className="text-gray-600">
                You can download the detailed report by clicking the download
                icon next to the result. The report will include the prediction,
                confidence score, and any other relevant information.
              </p>
            </div>
          </div>

          <div className="flex items-start space-x-4">
            <div className="flex items-center justify-center w-12 h-12  text-black rounded-xl">
              <FaArrowCircleRight size={24} />
            </div>
            <div>
              <h3 className="text-2xl font-semibold text-gray-800 mb-2">
                Step 5: View Medical History
              </h3>
              <p className="text-gray-600">
                You can access your past predictions and reports by navigating
                to the Medical History page. All your previous X-ray image
                results will be listed there, and you can download any report
                again.
              </p>
            </div>
          </div>

          <div className="flex items-start space-x-4">
            <div className="flex items-center justify-center w-12 h-12  text-black rounded-xl">
              <FaArrowCircleRight size={24} />
            </div>
            <div>
              <h3 className="text-2xl font-semibold text-gray-800 mb-2">
                Step 6: Clear Database (Admin Only)
              </h3>
              <p className="text-gray-600">
                If you have administrative access, you can clear the entire
                database to reset all reports. This can be done from the Medical
                History page by clicking the Clear All Reports button.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
