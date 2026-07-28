import Link from "next/link";

interface Props {
  id: string;
  foodType?: string;
  freshnessScore?: number;
  status: string;
  createdAt: string;
}

export function InspectionCard({ id, foodType, freshnessScore, status, createdAt }: Props) {
  const statusColor =
    status === "completed" ? "text-green-600 bg-green-50" :
    status === "processing" ? "text-yellow-600 bg-yellow-50" :
    status === "failed" ? "text-red-600 bg-red-50" :
    "text-gray-600 bg-gray-50";

  return (
    <Link href={`/inspection/${id}`} className="block bg-white rounded-lg shadow hover:shadow-md transition">
      <div className="p-4 flex items-center justify-between">
        <div>
          <p className="font-medium text-gray-900">{foodType || "Unknown food"}</p>
          <p className="text-sm text-gray-500 mt-1">{new Date(createdAt).toLocaleDateString()}</p>
        </div>
        <div className="text-right">
          {freshnessScore !== undefined && (
            <p className="text-lg font-bold text-gray-900">{freshnessScore}%</p>
          )}
          <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${statusColor}`}>
            {status}
          </span>
        </div>
      </div>
    </Link>
  );
}
