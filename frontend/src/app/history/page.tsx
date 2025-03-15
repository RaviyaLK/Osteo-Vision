export default function HistoryPage() {
  return (
    <div className="relative min-h-screen bg-gray-100 p-8 flex items-center justify-center">
      {/* Background Image */}
      <div className="absolute inset-0 bg-cover bg-center opacity-40" style={{ backgroundImage: 'url("/background.jpg")' }}></div>
      
      <div className="relative z-10 container mx-auto p-6 bg-white shadow-md rounded-lg">
        <h1 className="text-3xl font-bold mb-4">Medical History</h1>
        <p>Your medical history and previous diagnoses will be shown here.</p>
      </div>
    </div>
  );
}
