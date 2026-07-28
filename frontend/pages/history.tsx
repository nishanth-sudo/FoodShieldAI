import { useEffect, useState } from "react";
import { ProtectedRoute } from "components/ProtectedRoute";
import { inspectionApi } from "lib/api";
import { InspectionCard } from "components/InspectionCard";
import { LoadingSpinner } from "components/LoadingSpinner";

export default function HistoryPage() {
  return (
    <ProtectedRoute>
      <HistoryContent />
    </ProtectedRoute>
  );
}

function HistoryContent() {
  const [inspections, setInspections] = useState<any[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const limit = 20;

  useEffect(() => {
    setLoading(true);
    inspectionApi.list(page, limit).then(({ data }) => {
      setInspections(data.items);
      setTotal(data.total);
    }).finally(() => setLoading(false));
  }, [page]);

  const totalPages = Math.ceil(total / limit);

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900 mb-6">Inspection History</h1>

      {loading ? (
        <LoadingSpinner text="Loading history..." />
      ) : inspections.length === 0 ? (
        <div className="text-center py-12 text-gray-500">
          <p className="text-4xl mb-3">📋</p>
          <p>No inspections yet. Start by uploading a food image.</p>
        </div>
      ) : (
        <>
          <div className="space-y-3">
            {inspections.map((item) => (
              <InspectionCard
                key={item.id}
                id={item.id}
                foodType={item.food_type}
                freshnessScore={item.freshness_score}
                status={item.status}
                createdAt={item.created_at}
              />
            ))}
          </div>

          {totalPages > 1 && (
            <div className="flex justify-center items-center gap-4 mt-8">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page <= 1}
                className="px-3 py-1.5 text-sm border rounded-lg disabled:opacity-30 hover:bg-gray-50"
              >
                Previous
              </button>
              <span className="text-sm text-gray-500">
                Page {page} of {totalPages}
              </span>
              <button
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page >= totalPages}
                className="px-3 py-1.5 text-sm border rounded-lg disabled:opacity-30 hover:bg-gray-50"
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
