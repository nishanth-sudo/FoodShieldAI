import { useEffect, useState } from "react";
import { ProtectedRoute } from "components/ProtectedRoute";
import { inspectionApi } from "lib/api";
import { InspectionCard } from "components/InspectionCard";
import { LoadingSpinner } from "components/LoadingSpinner";
import { useLanguage } from "context/LanguageContext";
import { useInspection } from "context/InspectionContext";
import type { Inspection } from "lib/types";

export default function HistoryPage() {
  return (
    <ProtectedRoute>
      <HistoryContent />
    </ProtectedRoute>
  );
}

function HistoryContent() {
  const [inspections, setInspections] = useState<Inspection[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const { t } = useLanguage();
  const { isOnline, localHistory, exportInspectionAsJson } = useInspection();
  const limit = 20;

  useEffect(() => {
    if (isOnline) {
      setLoading(true);
      inspectionApi
        .list(page, limit)
        .then(({ data }) => {
          setInspections(data.items);
          setTotal(data.total);
        })
        .catch((err) => {
          console.error("Failed to load history online, falling back to local storage", err);
          setInspections(localHistory);
          setTotal(localHistory.length);
        })
        .finally(() => setLoading(false));
    } else {
      // offline fallback
      setInspections(localHistory);
      setTotal(localHistory.length);
      setLoading(false);
    }
  }, [page, isOnline, localHistory]);

  const totalPages = Math.ceil(total / limit);

  const handleExportAll = () => {
    if (inspections.length === 0) return;
    const jsonString = `data:text/json;charset=utf-8,${encodeURIComponent(
      JSON.stringify(inspections, null, 2)
    )}`;
    const downloadAnchor = document.createElement("a");
    downloadAnchor.setAttribute("href", jsonString);
    downloadAnchor.setAttribute("download", `foodshield_history_export.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  return (
    <div className="animate-fadeIn space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">{t.history}</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">{t.historyDesc}</p>
        </div>
        {inspections.length > 0 && (
          <button
            onClick={handleExportAll}
            className="inline-flex items-center gap-2 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 border border-gray-300 dark:border-gray-600 px-4 py-2 rounded-lg text-sm font-semibold hover:bg-gray-50 dark:hover:bg-gray-700 transition focus:outline-none focus:ring-2 focus:ring-primary-500 shadow-sm"
            aria-label={t.exportData}
          >
            📥 {t.exportData}
          </button>
        )}
      </div>

      {loading ? (
        <LoadingSpinner text={t.loading} />
      ) : inspections.length === 0 ? (
        <div className="text-center py-16 bg-white dark:bg-gray-800 rounded-xl border border-gray-100 dark:border-gray-700 shadow-sm text-gray-500 dark:text-gray-400">
          <p className="text-5xl mb-4">📋</p>
          <p className="font-semibold text-lg">{t.noInspections}</p>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 gap-4" role="list">
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

          {isOnline && totalPages > 1 && (
            <div className="flex justify-center items-center gap-4 mt-8">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page <= 1}
                className="px-4 py-2 text-sm font-semibold border rounded-lg bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 border-gray-300 dark:border-gray-600 disabled:opacity-30 hover:bg-gray-50 dark:hover:bg-gray-700 transition focus:outline-none focus:ring-2 focus:ring-primary-500"
                aria-label={t.previous}
              >
                {t.previous}
              </button>
              <span className="text-sm text-gray-500 dark:text-gray-400">
                {t.pageOf.replace("{page}", page.toString()).replace("{totalPages}", totalPages.toString())}
              </span>
              <button
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page >= totalPages}
                className="px-4 py-2 text-sm font-semibold border rounded-lg bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 border-gray-300 dark:border-gray-600 disabled:opacity-30 hover:bg-gray-50 dark:hover:bg-gray-700 transition focus:outline-none focus:ring-2 focus:ring-primary-500"
                aria-label={t.next}
              >
                {t.next}
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
