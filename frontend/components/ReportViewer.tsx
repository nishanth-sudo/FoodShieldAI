import { useState } from "react";
import { useLanguage } from "../context/LanguageContext";

interface Props {
  report?: string | null;
}

export function ReportViewer({ report }: Props) {
  const { t } = useLanguage();
  const [isExpanded, setIsExpanded] = useState(true);

  if (!report) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-5">
        <h3 className="text-sm font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider mb-2">
          {t.inspectionReport}
        </h3>
        <p className="text-gray-400 dark:text-gray-500 text-sm italic">{t.noReport}</p>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-5 overflow-hidden">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider">
          {t.inspectionReport}
        </h3>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="text-xs text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 font-semibold focus:outline-none focus:ring-2 focus:ring-primary-500 rounded px-1"
          aria-expanded={isExpanded}
          aria-label={`${isExpanded ? t.collapse : t.expand} inspection report`}
        >
          {isExpanded ? t.collapse : t.expand}
        </button>
      </div>

      {isExpanded && (
        <div className="prose prose-sm max-w-none text-gray-600 dark:text-gray-300 whitespace-pre-wrap leading-relaxed animate-fadeIn">
          {report}
        </div>
      )}
    </div>
  );
}
export default ReportViewer;
