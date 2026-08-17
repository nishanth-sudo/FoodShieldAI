import { useRouter } from "next/router";
import { useEffect, useState } from "react";
import { ProtectedRoute } from "components/ProtectedRoute";
import { inspectionApi } from "lib/api";
import { CONFIG } from "../../lib/config";
import { FreshnessGauge } from "components/FreshnessGauge";
import { XAIViewer } from "components/XAIViewer";
import { ReportViewer } from "components/ReportViewer";
import { LoadingSpinner } from "components/LoadingSpinner";
import { useLanguage } from "../../context/LanguageContext";
import { useInspection } from "../../context/InspectionContext";
import { getConfidence } from "../../lib/types";
import type { Inspection } from "../../lib/types";

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
  const [inspection, setInspection] = useState<Inspection | null>(null);
  const [loading, setLoading] = useState(true);
  const { t } = useLanguage();
  const { isOnline, localHistory, exportInspectionAsJson, deleteLocalInspection } = useInspection();

  useEffect(() => {
    if (!id) return;
    setLoading(true);

    if (isOnline) {
      inspectionApi
        .get(id as string)
        .then(({ data }) => {
          setInspection(data);
        })
        .catch((err) => {
          console.error("Failed to load details online, trying local cache", err);
          const localItem = localHistory.find((item) => item.id === id);
          if (localItem) setInspection(localItem);
        })
        .finally(() => setLoading(false));
    } else {
      const localItem = localHistory.find((item) => item.id === id);
      if (localItem) {
        setInspection(localItem);
      }
      setLoading(false);
    }
  }, [id, isOnline, localHistory]);

  const handleDelete = () => {
    if (!id) return;
    if (confirm(t.confirmDelete)) {
      deleteLocalInspection(id as string);
      router.push("/history");
    }
  };

  if (loading) return <LoadingSpinner text={t.loading} />;
  if (!inspection) {
    return (
      <div className="text-center py-16 text-gray-500 dark:text-gray-400">
        <p className="text-5xl mb-4">🔍</p>
        <p className="text-lg font-bold">{t.inspectionNotFound}</p>
        <button
          onClick={() => router.push("/history")}
          className="mt-4 text-primary-600 dark:text-primary-400 hover:underline font-semibold"
        >
          {t.returnToHistory}
        </button>
      </div>
    );
  }

  const isLowConfidence =
    inspection.freshness_score != null &&
    inspection.freshness_score < CONFIG.CONFIDENCE_THRESHOLD * 100;
  const confidence = getConfidence(inspection);
  const defects = inspection.packaging_defects ?? [];

  return (
    <div className="animate-fadeIn space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <button
          onClick={() => router.back()}
          className="text-sm font-semibold text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 focus:outline-none focus:ring-2 focus:ring-primary-500 rounded px-1 py-0.5"
          aria-label={t.back}
        >
          &larr; {t.back}
        </button>

        <div className="flex items-center gap-3">
          <button
            onClick={() => exportInspectionAsJson(inspection)}
            className="px-4 py-2 border rounded-lg text-sm font-bold bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700 transition focus:outline-none focus:ring-2 focus:ring-primary-500 shadow-sm"
          >
            📥 {t.exportData}
          </button>
          <button
            onClick={handleDelete}
            className="px-4 py-2 border rounded-lg text-sm font-bold bg-red-50 dark:bg-red-950/20 text-red-600 dark:text-red-400 border-red-200 dark:border-red-800 hover:bg-red-100 dark:hover:bg-red-900/30 transition focus:outline-none focus:ring-2 focus:ring-red-500 shadow-sm"
          >
            🗑️ {t.delete}
          </button>
        </div>
      </div>

      <h1 className="text-3xl font-black text-gray-900 dark:text-white">
        {inspection.food_type || t.unknownFood}
      </h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
        {/* Left Side: Overview & Visual Details */}
        <div className="space-y-6">
          {isLowConfidence && (
            <div
              role="alert"
              className="bg-amber-50 dark:bg-amber-950/20 text-amber-800 dark:text-amber-400 border border-amber-200 dark:border-amber-800 p-4 rounded-xl text-sm leading-relaxed"
            >
              ⚠️ {t.lowConfidenceWarning}
            </div>
          )}

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <FreshnessGauge score={inspection.freshness_score || 0} label={t.freshness} />

            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-5 flex flex-col justify-between">
              <div>
                <p className="text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
                  {t.status}
                </p>
                <span
                  className={`inline-block px-3 py-1 rounded-full text-xs font-bold border ${
                    inspection.status === "completed"
                      ? "text-green-700 bg-green-50 border-green-200 dark:text-green-400 dark:bg-green-900/20 dark:border-green-800"
                      : "text-red-700 bg-red-50 border-red-200 dark:text-red-400 dark:bg-red-900/20 dark:border-red-800"
                  }`}
                >
                  {t[inspection.status as keyof typeof t] || inspection.status}
                </span>
              </div>
              {confidence !== undefined && (
                <div className="mt-4 pt-4 border-t border-gray-100 dark:border-gray-700">
                  <p className="text-xs text-gray-400 dark:text-gray-500 font-medium mb-1">
                    {t.confidence}: {(confidence * 100).toFixed(0)}%
                  </p>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 h-1.5 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-primary-500"
                      style={{ width: `${(confidence * 100).toFixed(0)}%` }}
                    />
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-5">
              <p className="text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">
                {t.shelfLife}
              </p>
              <p className="text-2xl font-black text-gray-900 dark:text-white">
                {inspection.shelf_life_days ?? "—"}{" "}
                <span className="text-sm font-normal text-gray-400 dark:text-gray-500">{t.days}</span>
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-5">
              <p className="text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">
                {t.defects}
              </p>
              <p className="text-2xl font-black text-gray-900 dark:text-white">
                {defects.length}
              </p>
            </div>
          </div>

          {/* Heatmap overlay image */}
          {inspection.image_url ? (
            <XAIViewer originalUrl={inspection.image_url} heatmapUrl={inspection.xai_heatmap_url} />
          ) : (
            <XAIViewer heatmapUrl={inspection.xai_heatmap_url} />
          )}
        </div>

        {/* Right Side: Defect Types, Risks and Reports */}
        <div className="space-y-6">
          {defects.length > 0 && (
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-5">
              <h3 className="text-sm font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-3">
                {t.packagingDefects}
              </h3>
              <ul className="space-y-2">
                {defects.map((def, idx) => (
                  <li
                    key={idx}
                    className="text-sm flex justify-between items-center py-1.5 border-b border-gray-50 dark:border-gray-700/50 last:border-0"
                  >
                    <span className="font-semibold text-gray-800 dark:text-gray-200">{def.type}</span>
                    <span className="text-xs bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 px-2 py-0.5 rounded font-bold">
                      {(def.confidence * 100).toFixed(0)}%
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {inspection.contamination_risks && Object.keys(inspection.contamination_risks).length > 0 && (
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-5">
              <h3 className="text-sm font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-3">
                {t.contaminationRisks}
              </h3>
              <div className="space-y-3">
                {Object.entries(inspection.contamination_risks).map(([riskType, val]) => {
                  const pct = ((val as number) * 100).toFixed(0);
                  return (
                    <div key={riskType} className="space-y-1">
                      <div className="flex justify-between text-xs font-semibold text-gray-600 dark:text-gray-400">
                        <span>{riskType}</span>
                        <span>{pct}%</span>
                      </div>
                      <div className="w-full bg-gray-100 dark:bg-gray-700 h-2 rounded-full overflow-hidden">
                        <div
                          className={`h-full ${
                            (val as number) > 0.5 ? "bg-red-500" : (val as number) > 0.2 ? "bg-amber-500" : "bg-green-500"
                          }`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          <ReportViewer report={inspection.report} />
        </div>
      </div>
    </div>
  );
}
