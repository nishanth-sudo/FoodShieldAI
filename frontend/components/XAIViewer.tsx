import { useState } from "react";

interface Props {
  originalUrl?: string;
  heatmapUrl?: string;
}

export function XAIViewer({ originalUrl, heatmapUrl }: Props) {
  const [showHeatmap, setShowHeatmap] = useState(false);

  if (!heatmapUrl && !originalUrl) {
    return <p className="text-gray-400 text-sm">No image available</p>;
  }

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-700">AI Focus Map</h3>
        {heatmapUrl && (
          <button
            onClick={() => setShowHeatmap(!showHeatmap)}
            className="text-xs text-primary-600 hover:text-primary-700"
          >
            {showHeatmap ? "Show original" : "Show AI focus"}
          </button>
        )}
      </div>
      <div className="relative aspect-video bg-gray-100 rounded-lg overflow-hidden">
        <img
          src={showHeatmap && heatmapUrl ? heatmapUrl : originalUrl}
          alt="Inspection"
          className="w-full h-full object-contain"
        />
      </div>
      {showHeatmap && (
        <p className="text-xs text-gray-400 mt-2">
          Red areas indicate regions that influenced the AI&apos;s decision
        </p>
      )}
    </div>
  );
}
