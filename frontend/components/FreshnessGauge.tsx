interface Props {
  score: number;
  label: string;
}

export function FreshnessGauge({ score, label }: Props) {
  const color = score >= 80 ? "bg-green-500" : score >= 50 ? "bg-yellow-500" : "bg-red-500";
  return (
    <div className="bg-white rounded-lg shadow p-4">
      <p className="text-sm text-gray-500 mb-2">{label}</p>
      <div className="flex items-end gap-3">
        <span className="text-3xl font-bold">{score}</span>
        <span className="text-lg text-gray-400">/ 100</span>
      </div>
      <div className="mt-2 h-2 bg-gray-200 rounded-full">
        <div className={`h-2 rounded-full ${color}`} style={{ width: `${score}%` }} />
      </div>
    </div>
  );
}
