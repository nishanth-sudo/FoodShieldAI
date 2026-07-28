import { useState, useCallback } from "react";
import { useRouter } from "next/router";
import { ProtectedRoute } from "components/ProtectedRoute";
import { inspectionApi } from "lib/api";
import { FreshnessGauge } from "components/FreshnessGauge";
import { XAIViewer } from "components/XAIViewer";
import { ReportViewer } from "components/ReportViewer";
import { LoadingSpinner } from "components/LoadingSpinner";

export default function InspectPage() {
  return (
    <ProtectedRoute>
      <InspectContent />
    </ProtectedRoute>
  );
}

function InspectContent() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState("");
  const router = useRouter();

  const onFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError("");
  }, []);

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError("");
    try {
      const { data } = await inspectionApi.upload(file);
      const pollInterval = setInterval(async () => {
        try {
          const { data: updated } = await inspectionApi.get(data.id);
          if (updated.status === "completed" || updated.status === "failed") {
            clearInterval(pollInterval);
            setResult(updated);
            setUploading(false);
          }
        } catch {
          clearInterval(pollInterval);
          setUploading(false);
        }
      }, 2000);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Upload failed");
      setUploading(false);
    }
  };

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900 mb-6">Food Inspection</h1>

      {!preview && (
        <div className="bg-white rounded-lg shadow p-8 text-center">
          <label className="cursor-pointer flex flex-col items-center gap-3">
            <div className="text-5xl text-gray-300">📷</div>
            <p className="text-gray-600">Select a food image to inspect</p>
            <span className="inline-block bg-primary-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-primary-700">
              Choose Image
            </span>
            <input type="file" accept="image/jpeg,image/png,image/webp" onChange={onFileSelect} className="hidden" />
          </label>
        </div>
      )}

      {preview && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg shadow p-4">
            <img src={preview} alt="Preview" className="w-full rounded-lg object-contain max-h-96" />
            <div className="mt-4 flex gap-3">
              <button
                onClick={handleUpload}
                disabled={uploading}
                className="bg-primary-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-primary-700 disabled:opacity-50"
              >
                {uploading ? "Analyzing..." : "Run Inspection"}
              </button>
              <button
                onClick={() => { setFile(null); setPreview(null); }}
                className="text-gray-600 px-4 py-2 rounded-lg text-sm border hover:bg-gray-50"
              >
                Cancel
              </button>
            </div>
          </div>

          {uploading && <LoadingSpinner text="AI is analyzing your food image..." />}

          {error && (
            <div className="bg-red-50 text-red-700 p-4 rounded-lg text-sm">{error}</div>
          )}

          {result && !uploading && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <FreshnessGauge score={result.freshness_score || 0} label="Freshness" />
                <div className="bg-white rounded-lg shadow p-4">
                  <p className="text-sm text-gray-500">Food Type</p>
                  <p className="text-xl font-bold text-gray-900">{result.food_type || "Unknown"}</p>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-white rounded-lg shadow p-4">
                  <p className="text-sm text-gray-500">Shelf Life</p>
                  <p className="text-xl font-bold text-gray-900">
                    {result.shelf_life_days ?? "—"} <span className="text-sm font-normal text-gray-500">days</span>
                  </p>
                </div>
                <div className="bg-white rounded-lg shadow p-4">
                  <p className="text-sm text-gray-500">Defects</p>
                  <p className="text-xl font-bold text-gray-900">{result.packaging_defects?.length || 0}</p>
                </div>
              </div>
              <XAIViewer originalUrl={preview} heatmapUrl={result.xai_heatmap_url} />
              <ReportViewer report={result.report} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
