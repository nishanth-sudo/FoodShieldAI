import { useAuth } from "context/AuthContext";
import Link from "next/link";

export default function HomePage() {
  const { user } = useAuth();

  if (!user) {
    return (
      <div className="text-center py-20">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">FoodShield AI</h1>
        <p className="text-lg text-gray-600 mb-8">AI-Powered Food Quality Inspection Platform</p>
        <Link href="/login" className="inline-block bg-primary-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-primary-700">
          Get Started
        </Link>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Welcome, {user.name}</h1>
        <p className="text-gray-500 mt-1">Upload a food image to start an inspection</p>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Link href="/inspect" className="bg-white rounded-lg shadow p-6 hover:shadow-md transition">
          <div className="text-3xl mb-3">📷</div>
          <h2 className="font-semibold text-gray-900">New Inspection</h2>
          <p className="text-sm text-gray-500 mt-1">Upload or capture a food image for AI analysis</p>
        </Link>
        <Link href="/history" className="bg-white rounded-lg shadow p-6 hover:shadow-md transition">
          <div className="text-3xl mb-3">📋</div>
          <h2 className="font-semibold text-gray-900">History</h2>
          <p className="text-sm text-gray-500 mt-1">View your past inspection results</p>
        </Link>
        {user.role === "admin" && (
          <Link href="/admin" className="bg-white rounded-lg shadow p-6 hover:shadow-md transition">
            <div className="text-3xl mb-3">⚙️</div>
            <h2 className="font-semibold text-gray-900">Admin</h2>
            <p className="text-sm text-gray-500 mt-1">Manage users and monitor system health</p>
          </Link>
        )}
      </div>
    </div>
  );
}
