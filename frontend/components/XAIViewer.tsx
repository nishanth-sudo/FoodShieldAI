import { useState } from "react";
import { useLanguage } from "../context/LanguageContext";

interface Props {
  originalUrl?: string;
  heatmapUrl?: string;
}

export function XAIViewer({ originalUrl, heatmapUrl }: Props) {
  const [showHeatmap, setShowHeatmap] = useState(false);
  const { t } = useLanguage();

  if (!heatmapUrl && !originalUrl) {
    return <p className="text-gray-400 dark:text-gray-500 text-sm italic">No image available</p>;
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider">
          {t.xaiMap}
        </h3>
        {heatmapUrl && (
          <button
            onClick={() => setShowHeatmap(!showHeatmap)}
            className="text-xs font-semibold text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 focus:outline-none focus:ring-2 focus:ring-primary-500 rounded px-1.5 py-0.5 border border-primary-200 dark:border-primary-800 bg-primary-50/50 dark:bg-primary-950/20"
            aria-label={showHeatmap ? t.showOriginal : t.showAI}
          >
            {showHeatmap ? t.showOriginal : t.showAI}
          </button>
        )}
      </div>

      <div className="relative aspect-video bg-gray-50 dark:bg-gray-900/50 rounded-lg overflow-hidden border border-gray-100 dark:border-gray-800">
        <img
          src={showHeatmap && heatmapUrl ? heatmapUrl : originalUrl}
          alt={showHeatmap ? "AI Explainability Heatmap" : "Original Food Inspection Image"}
          className="w-full h-full object-contain"
          loading="lazy"
        />
      </div>

      {showHeatmap && (
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-3 leading-relaxed">
          💡 {t.xaiExplanation}
        </p>
      )}
    </div>
  );
}
export default XAIViewer;
