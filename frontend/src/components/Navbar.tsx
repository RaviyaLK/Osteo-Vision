/* eslint-disable @next/next/no-img-element */
"use client"; // Required for handling state in Next.js

import { useState } from "react";
import Link from "next/link";
import { AiOutlineMenu, AiOutlineClose } from "react-icons/ai"; // Import icons

export default function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <nav className="bg-blue-600 text-white shadow-md fixed w-full top-0 z-50">
      <div className="container mx-auto px-4 lg:px-6 py-3 flex justify-between items-center">

        {/* Logo */}
        <Link
          href="/"
          className="text-3xl font-bold flex items-center space-x-2"
        >
          <img src="./favicon.ico" alt="Logo" className="h-14 w-14" />
          <span>Osteo-Vision</span>
        </Link>

        {/* Desktop Menu */}
        <div className="hidden md:flex space-x-6">
          <Link href="/" className="hover:text-blue-200 text-xl ">
            Home
          </Link>
          <Link href="/report" className="hover:text-blue-200 text-xl">
            Report
          </Link>
          <Link href="/history" className="hover:text-blue-200 text-xl">
            History
          </Link>
          <Link href="/documentation" className="hover:text-blue-200 text-xl">
            Documentation
          </Link>
          
        </div>

        {/* Mobile Menu Button */}
        <button
          className="md:hidden text-2xl focus:outline-hidden"
          onClick={() => setMenuOpen(!menuOpen)}
        >
          {menuOpen ? <AiOutlineClose /> : <AiOutlineMenu />}
        </button>
      </div>

      {/* Mobile Menu */}
      {menuOpen && (
        <div className="md:hidden bg-blue-700 py-2">
          <Link href="/" className="block px-4 py-2 hover:bg-blue-800">
            Home
          </Link>
          <Link href="/report" className="block px-4 py-2 hover:bg-blue-800">
            Report
          </Link>
          <Link href="/history" className="block px-4 py-2 hover:bg-blue-800">
            History
          </Link>
          <Link
            href="/documentation"
            className="block px-4 py-2 hover:bg-blue-800"
          >
            Documentation
          </Link>
        </div>
      )}
    </nav>
  );
}
