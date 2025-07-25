"use client";
import Link from "next/link";
import { FaClinicMedical, FaChartLine} from "react-icons/fa";
import { motion } from "framer-motion";

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white">
      
      <div className="relative overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24 md:py-32">
          <div className="relative z-10 text-center">
            <motion.h1 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              className="text-4xl md:text-6xl font-bold text-gray-900 mb-6"
            >
              Advanced Knee Osteoporosis Detection
            </motion.h1>
            
            <motion.p 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="text-xl md:text-2xl text-gray-600 max-w-3xl mx-auto mb-10"
            >
              AI-powered analysis for early detection and accurate diagnosis of knee osteoporosis
            </motion.p>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.4 }}
            >
              <Link
                href="/prediction"
                className="inline-flex items-center px-8 py-4 border border-transparent text-xl font-medium rounded-xl shadow-lg text-white bg-blue-600 hover:bg-blue-700 transition-all duration-300 transform hover:scale-105"
              >
                Get Started with Analysis
                <svg className="ml-3 -mr-1 h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M12.293 5.293a1 1 0 011.414 0l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-2.293-2.293a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
              </Link>
            </motion.div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-extrabold text-gray-900 sm:text-4xl">
              Why Choose Osteo-Vision?
            </h2>
            <p className="mt-4 max-w-2xl text-xl text-gray-500 mx-auto">
              Cutting-edge technology for precise bone health assessment
            </p>
          </div>

          <div className="grid grid-cols-1 gap-12 md:grid-cols-2 lg:grid-cols-2">
          
            <motion.div 
              whileHover={{ y: -10 }}
              className="bg-blue-50 p-8 rounded-xl shadow-md"
            >
              <div className="flex items-center justify-center h-12 w-12 rounded-md bg-blue-600 text-white mb-4">
                <FaClinicMedical className="h-6 w-6" />
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">Clinical Accuracy</h3>
              <p className="text-gray-600">
               More than 95% accuracy in detecting early signs of osteoporosis from knee X-rays
              </p>
            </motion.div>

         
            <motion.div 
              whileHover={{ y: -10 }}
              className="bg-blue-50 p-8 rounded-xl shadow-md"
            >
              <div className="flex items-center justify-center h-12 w-12 rounded-md bg-blue-600 text-white mb-4">
                <FaChartLine className="h-6 w-6" />
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">Advanced Analytics</h3>
              <p className="text-gray-600">
                Detailed visual explanations showing bone density variations
              </p>
            </motion.div>

           
           

           
          </div>
        </div>
      </div>

      {/* How It Works */}
      <div className="py-16 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="lg:text-center">
            <h2 className="text-3xl font-extrabold text-gray-900 sm:text-4xl">
              How Our System Works
            </h2>
            <p className="mt-4 max-w-2xl text-xl text-gray-500 lg:mx-auto">
              Simple three-step process for accurate diagnosis
            </p>
          </div>

          <div className="mt-20">
            <div className="space-y-10 md:space-y-0 md:grid md:grid-cols-3 md:gap-x-8 md:gap-y-10">
              {/* Step 1 */}
              <div className="relative group">
                <div className="absolute -left-4 -top-4 h-16 w-16 rounded-full bg-blue-100 opacity-70 group-hover:opacity-100 transition-all duration-300"></div>
                <div className="relative">
                  <div className="flex items-center justify-center h-12 w-12 rounded-md bg-blue-600 text-white">
                    <span className="text-xl font-bold">1</span>
                  </div>
                  <h3 className="mt-4 text-lg font-medium text-gray-900">Upload X-ray</h3>
                  <p className="mt-2 text-gray-600">
                    Simply drag and drop your knee X-ray image in  JPEG, or PNG format
                  </p>
                </div>
              </div>

              {/* Step 2 */}
              <div className="relative group">
                <div className="absolute -left-4 -top-4 h-16 w-16 rounded-full bg-blue-100 opacity-70 group-hover:opacity-100 transition-all duration-300"></div>
                <div className="relative">
                  <div className="flex items-center justify-center h-12 w-12 rounded-md bg-blue-600 text-white">
                    <span className="text-xl font-bold">2</span>
                  </div>
                  <h3 className="mt-4 text-lg font-medium text-gray-900">AI Analysis</h3>
                  <p className="mt-2 text-gray-600">
                    Our deep learning models analyze bone density patterns in seconds
                  </p>
                </div>
              </div>

              {/* Step 3 */}
              <div className="relative group">
                <div className="absolute -left-4 -top-4 h-16 w-16 rounded-full bg-blue-100 opacity-70 group-hover:opacity-100 transition-all duration-300"></div>
                <div className="relative">
                  <div className="flex items-center justify-center h-12 w-12 rounded-md bg-blue-600 text-white">
                    <span className="text-xl font-bold">3</span>
                  </div>
                  <h3 className="mt-4 text-lg font-medium text-gray-900">Get Results</h3>
                  <p className="mt-2 text-gray-600">
                    Receive detailed report with diagnosis, confidence score, and visual explanations
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      
      <div className="bg-blue-600">
        <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:py-16 lg:px-8 lg:flex lg:items-center lg:justify-between">
          <h2 className="text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
            <span className="block">Ready to analyze your X-rays?</span>
            <span className="block text-blue-200">Get started today.</span>
          </h2>
          <div className="mt-8 flex lg:mt-0 lg:flex-shrink-0">
            <div className="inline-flex rounded-xl shadow">
              <Link
                href="/prediction"
                className="inline-flex items-center justify-center px-8 py-4 border border-transparent text-xl font-medium rounded-xl text-white bg-blue-800 hover:bg-blue-700 transition-all duration-300 transform hover:scale-105"
              >
                Start Free Analysis
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}