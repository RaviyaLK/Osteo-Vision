import Link from 'next/link'

export default function Navbar() {
  return (
    <nav className="bg-blue-600 p-4 text-white">
      <div className="container mx-auto flex justify-between items-center">
        <Link href="/" className="text-2xl font-bold">Osteo-Vision</Link>
        <div className="space-x-4">
          <Link href="/" className="hover:text-blue-200">Home</Link>
          <Link href="/report" className="hover:text-blue-200">Report</Link>
          <Link href="/history" className="hover:text-blue-200">History</Link>
        </div>
      </div>
    </nav>
  )
}