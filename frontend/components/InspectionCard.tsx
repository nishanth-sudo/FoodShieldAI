import Link from "next/link";
import { useLanguage } from "context/LanguageContext";

interface Props {
  id: string;
  foodType?: string | null;
  freshnessScore?: number | null;
  status: string;
  createdAt?: string | null;
}

export function InspectionCard({ id, foodType, freshnessScore, status, createdAt }: Props) {
  const { t } = useLanguage();

  const statusColors = {
    completed: "text-green-700 bg-green-50 border-green-200 dark:text-green-400 dark:bg-green-900/20 dark:border-green-800",
    processing: "text-amber-700 bg-amber-50 border-amber-200 dark:text-amber-400 dark:bg-amber-900/20 dark:border-amber-800",
    failed: "text-red-700 bg-red-50 border-red-200 dark:text-red-400 dark:bg-red-900/20 dark:border-red-800",
  };

  const currentStatusColor =
    statusColors[status as keyof typeof statusColors] ||
    "text-gray-700 bg-gray-50 border-gray-200 dark:text-gray-400 dark:bg-gray-900/20 dark:border-gray-800";

  // Translate status text safely
  const translatedStatus = t[status as keyof typeof t] || status;

  return (
    <Link
      href={`/inspection/${id}`}
      className="block bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-5 hover:shadow-md hover:border-primary-500 dark:hover:border-primary-500 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2"
      aria-label={`Inspection card for ${foodType || t.unknownFood}, Status: ${translatedStatus}, Freshness: ${
        freshnessScore !== undefined ? freshnessScore + "%" : "N/A"
      }`}
      role="listitem"
    >
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <p className="font-bold text-gray-900 dark:text-white text-base md:text-lg">
            {foodType || t.unknownFood}
          </p>
          <p className="text-xs md:text-sm text-gray-400 dark:text-gray-400">
            {createdAt ? new Date(createdAt).toLocaleString() : ""}
          </p>
        </div>
        <div className="text-right flex flex-col items-end gap-1.5">
          {freshnessScore !== undefined && (
            <p className="text-lg md:text-xl font-extrabold text-gray-900 dark:text-white">
              {freshnessScore}%
            </p>
          )}
          <span className={`inline-block px-3 py-1 rounded-full text-xs font-semibold border ${currentStatusColor}`}>
            {translatedStatus}
          </span>
        </div>
      </div>
    </Link>
  );
}
