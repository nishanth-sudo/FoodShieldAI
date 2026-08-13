import { useCallback } from "react";
import { ProtectedRoute } from "components/ProtectedRoute";
import { useInspection } from "../context/InspectionContext";
import { useLanguage } from "../context/LanguageContext";
import { CONFIG } from "../lib/config";
import { FreshnessGauge } from "components/FreshnessGauge";
import { XAIViewer } from "components/XAIViewer";
import { ReportViewer } from "components/ReportViewer";
import { LoadingSpinner } from "components/LoadingSpinner";
import ImageUploader from "components/ImageUploader";

export default function InspectPage() {
  return (
    <ProtectedRoute>
      <InspectContent />
    </ProtectedRoute>
  );
}

function InspectContent() {
  const {
    file,
    preview,
    status,
    statusText,
    result,
    error,
    isOnline,
    clearFile,
    runInspection,
    exportInspectionAsJson,
  } = useInspection();

  const { t } = useLanguage();

  const handleRetry = useCallback(() => {
    runInspection();
  }, [runInspection]);

  // Translate client error codes or fallback
  const getErrorMessage = () => {
    if (!error) return "";
    if (error === "invalidFileType") return t.invalidFileType;
    if (error === "fileTooLarge") return t.fileTooLarge;
    if (error === "offlineWarning") return t.offlineWarning;
    return error;
  };

  const isUploadingOrAnalyzing = status === "uploading" || status === "analyzing";

  // Check if AI confidence is low
  const isLowConfidence = result && result.freshness_score !== undefined && result.freshness_score < CONFIG.CONFIDENCE_THRESHOLD * 100;

  return (
    <div className="animate-fadeIn max-w-5xl mx-auto space-y-6">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
        {t.foodInspection}
      </h1>

      {getErrorMessage() && (
        <div
          role="alert"
          aria-live="assertive"
          className="bg-red-50 dark:bg-red-900/30 text-red-700 dark:text-red-400 border border-red-200 dark:border-red-800 p-4 rounded-xl text-sm flex flex-col sm:flex-row sm:items-center justify-between gap-3"
        >
          <span>⚠️ {getErrorMessage()}</span>
          {status === "failed" && isOnline && file && (
            <button
              onClick={handleRetry}
              className="bg-red-600 hover:bg-red-700 text-white font-semibold px-4 py-1.5 rounded-lg text-xs transition active:scale-95"
            >
              {t.retry}
            </button>
          )}
        </div>
      )}

      {!preview && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-6 md:p-10">
          <ImageUploader />
        </div>
      )}

      {preview && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
          {/* File Preview & Actions Column */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-5 space-y-5">
            <div className="relative aspect-video rounded-lg overflow-hidden bg-gray-50 dark:bg-gray-900/50 border border-gray-100 dark:border-gray-800">
              <img src={preview} alt="Food Upload Preview" className="w-full h-full object-contain" />
            </div>

            {!isUploadingOrAnalyzing && !result && (
              <div className="flex gap-4">
                <button
                  onClick={runInspection}
                  disabled={!isOnline}
                  className="flex-1 bg-primary-600 hover:bg-primary-700 text-white font-semibold py-3 rounded-lg text-sm shadow-sm hover:shadow hover:scale-[1.01] active:scale-95 disabled:opacity-50 transition duration-150"
                  aria-label={t.runInspection}
                >
                  {t.runInspection}
                </button>
                <button
                  onClick={clearFile}
                  className="bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-200 font-semibold px-5 py-3 rounded-lg text-sm transition"
                  aria-label={t.cancel}
                >
                  {t.cancel}
                </button>
              </div>
            )}

            {isUploadingOrAnalyzing && (
              <div className="space-y-4">
                <LoadingSpinner text={statusText} />
                <div className="w-full bg-gray-200 dark:bg-gray-700 h-2.5 rounded-full overflow-hidden">
                  <div
                    className={`h-full bg-primary-600 transition-all duration-500 ${
                      status === "uploading" ? "w-1/3" : "w-2/3"
                    }`}
                  />
                </div>
              </div>
            )}

            {result && !isUploadingOrAnalyzing && (
              <div className="flex gap-4">
                <button
                  onClick={() => exportInspectionAsJson(result)}
                  className="flex-1 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700 font-semibold py-2.5 rounded-lg text-sm transition shadow-sm"
                >
                  📥 {t.exportData}
                </button>
                <button
                  onClick={clearFile}
                  className="bg-primary-600 hover:bg-primary-700 text-white font-semibold px-5 py-2.5 rounded-lg text-sm transition"
                >
                  {t.newInspection}
                </button>
              </div>
            )}
          </div>

          {/* Results Analysis Column */}
          {result && !isUploadingOrAnalyzing && (
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
                <FreshnessGauge score={result.freshness_score || 0} label={t.freshness} />

                <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-5 flex flex-col justify-between">
                  <div>
                    <p className="text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
                      {t.foodType}
                    </p>
                    <p className="text-2xl font-black text-gray-900 dark:text-white leading-tight">
                      {result.food_type || t.unknownFood}
                    </p>
                  </div>
                  {result.confidence !== undefined && (
                    <div className="mt-4 pt-4 border-t border-gray-100 dark:border-gray-700">
                      <p className="text-xs text-gray-400 dark:text-gray-500 font-medium mb-1">
                        {t.confidence}: {(result.confidence * 100).toFixed(0)}%
                      </p>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 h-1.5 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-primary-500"
                          style={{ width: `${(result.confidence * 100).toFixed(0)}%` }}
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
                    {result.shelf_life_days ?? "—"}{" "}
                    <span className="text-sm font-normal text-gray-400 dark:text-gray-500">days</span>
                  </p>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-5">
                  <p className="text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">
                    {t.defects}
                  </p>
                  <p className="text-2xl font-black text-gray-900 dark:text-white">
                    {result.packaging_defects?.length || 0}
                  </p>
                </div>
              </div>

              {result.packaging_defects?.length > 0 && (
                <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-5">
                  <h3 className="text-sm font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-3">
                    {t.packagingDefects}
                  </h3>
                  <ul className="space-y-2">
                    {result.packaging_defects.map((def: any, idx: number) => (
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

              {result.contamination_risks && Object.keys(result.contamination_risks).length > 0 && (
                <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-5">
                  <h3 className="text-sm font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-3">
                    {t.contaminationRisks}
                  </h3>
                  <div className="space-y-3">
                    {Object.entries(result.contamination_risks).map(([riskType, val]) => {
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

              <XAIViewer originalUrl={preview} heatmapUrl={result.xai_heatmap_url} />

              <ReportViewer report={result.report} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
