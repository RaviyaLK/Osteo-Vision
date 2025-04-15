/* eslint-disable @next/next/no-img-element */
"use client";

import { useState } from "react";
import Link from "next/link";
import { AiOutlineMenu, AiOutlineClose } from "react-icons/ai";
import { FiHome, FiFileText, FiClock, FiHelpCircle, FiZap } from "react-icons/fi";

export default function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <nav className="bg-blue-600 text-white shadow-sm border-b border-blue-700 fixed w-full top-0 z-60">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex-shrink-0 flex items-center">
            <Link href="/" className="flex items-center space-x-2">
              <img 
                src="/favicon.ico" 
                alt="Logo" 
                className="h-10 w-10 rounded-lg"
              />
              <span className="text-xl font-semibold">Osteo-Vision</span>
            </Link>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:block">
            <div className="ml-10 flex items-center space-x-8">
              <Link 
                href="/" 
                className="flex items-center text-blue-100 hover:text-white px-1 pt-1 text-sm font-medium transition-colors"
              >
                <FiHome className="mr-2" size={18} />
                Home
              </Link>
              <Link 
                href="/prediction" 
                className="flex items-center text-blue-100 hover:text-white px-1 pt-1 text-sm font-medium transition-colors"
              >
                <FiZap  className="mr-2" size={18} />
                Predict
              </Link>
              <Link 
                href="/report" 
                className="flex items-center text-blue-100 hover:text-white px-1 pt-1 text-sm font-medium transition-colors"
              >
                <FiFileText className="mr-2" size={18} />
                Report
              </Link>
              <Link 
                href="/history" 
                className="flex items-center text-blue-100 hover:text-white px-1 pt-1 text-sm font-medium transition-colors"
              >
                <FiClock className="mr-2" size={18} />
                History
              </Link>
              <Link 
                href="/documentation" 
                className="flex items-center text-blue-100 hover:text-white px-1 pt-1 text-sm font-medium transition-colors"
              >
                <FiHelpCircle className="mr-2" size={18} />
                Documentation
              </Link>
            </div>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden flex items-center">
            <button
              onClick={() => setMenuOpen(!menuOpen)}
              className="inline-flex items-center justify-center p-2 rounded-md text-blue-200 hover:text-white hover:bg-blue-700 focus:outline-none transition-colors"
            >
              {menuOpen ? (
                <AiOutlineClose size={20} />
              ) : (
                <AiOutlineMenu size={20} />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      {menuOpen && (
        <div className="md:hidden bg-blue-700 border-t border-blue-800">
          <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
            <Link
              href="/"
              className="flex items-center text-blue-100 hover:text-white hover:bg-blue-800 px-3 py-2 rounded-md text-sm font-medium transition-colors"
            >
              <FiHome className="mr-2" size={16} />
              Home
            </Link>
            <Link
              href="/report"
              className="flex items-center text-blue-100 hover:text-white hover:bg-blue-800 px-3 py-2 rounded-md text-sm font-medium transition-colors"
            >
              <FiFileText className="mr-2" size={16} />
              Report
            </Link>
            <Link
              href="/history"
              className="flex items-center text-blue-100 hover:text-white hover:bg-blue-800 px-3 py-2 rounded-md text-sm font-medium transition-colors"
            >
              <FiClock className="mr-2" size={16} />
              History
            </Link>
            <Link
              href="/documentation"
              className="flex items-center text-blue-100 hover:text-white hover:bg-blue-800 px-3 py-2 rounded-md text-sm font-medium transition-colors"
            >
              <FiHelpCircle className="mr-2" size={16} />
              Documentation
            </Link>
          </div>
        </div>
      )}
    </nav>
  );
}