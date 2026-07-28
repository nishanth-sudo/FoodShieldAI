import { useRouter } from "next/router";
import { useEffect, useState } from "react";
import { ProtectedRoute } from "components/ProtectedRoute";
import { inspectionApi } from "lib/api";
import { FreshnessGauge } from "components/FreshnessGauge";
import { XAIViewer } from "components/XAIViewer";
import { ReportViewer } from "components/ReportViewer";
import { LoadingSpinner } from "components/LoadingSpinner";

export default function InspectionDetailPage() {
  return (
    <ProtectedRoute>
      <DetailContent />
    </ProtectedRoute>
  );
}

function DetailContent() {
  const router = useRouter();
  const { id } = router.query;
  const [inspection, setInspection] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!id) return;
    setLoading(true);
    inspectionApi.get(id as string).then(({ data }) => {
      setInspection(data);
    }).finally(() => setLoading(false));
  }, [id]);

  if (loading) return <LoadingSpinner text="Loading inspection..." />;
  if (!inspection) return <p className="text-gray-500">Inspection not found.</p>;

  return (
    <div>
      <button onClick={() => router.back()} className="text-sm text-primary-600 hover:text-primary-700 mb-4">
        &larr; Back
      </button>
      <h1 className="text-2xl font-bold text-gray-900 mb-6">
        Inspection {inspection.food_type || "Result"}
      </h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <FreshnessGauge score={inspection.freshness_score || 0} label="Freshness" />
            <div className="bg-white rounded-lg shadow p-4">
              <p className="text-sm text-gray-500">Status</p>
              <span className={`inline-block mt-1 px-2 py-0.5 rounded-full text-xs font-medium
                ${inspection.status === "completed" ? "text-green-600 bg-green-50" : "text-gray-600 bg-gray-50"}`}>
                {inspection.status}
              </span>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-white rounded-lg shadow p-4">
              <p className="text-sm text-gray-500">Food Type</p>
              <p className="text-xl font-bold">{inspection.food_type || "Unknown"}</p>
            </div>
            <div className="bg-white rounded-lg shadow p-4">
              <p className="text-sm text-gray-500">Shelf Life</p>
              <p className="text-xl font-bold">{inspection.shelf_life_days ?? "—"} <span className="text-sm font-normal text-gray-500">days</span></p>
            </div>
          </div>
          <XAIViewer heatmapUrl={inspection.xai_heatmap_url} />
        </div>
        <div className="space-y-4">
          {inspection.packaging_defects?.length > 0 && (
            <div className="bg-white rounded-lg shadow p-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Packaging Defects</h3>
              <ul className="space-y-1">
                {inspection.packaging_defects.map((d: any, i: number) => (
                  <li key={i} className="text-sm text-gray-600">
                    {d.type} — {(d.confidence * 100).toFixed(0)}%
                  </li>
                ))}
              </ul>
            </div>
          )}
          {inspection.contamination_risks && Object.keys(inspection.contamination_risks).length > 0 && (
            <div className="bg-white rounded-lg shadow p-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Contamination Risks</h3>
              {Object.entries(inspection.contamination_risks).map(([key, val]) => (
                <div key={key} className="flex justify-between text-sm py-1">
                  <span className="text-gray-600">{key}</span>
                  <span className="font-medium">{((val as number) * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
          )}
          <ReportViewer report={inspection.report} />
        </div>
      </div>
    </div>
  );
}
