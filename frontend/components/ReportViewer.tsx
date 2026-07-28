interface Props {
  report: string;
}

export function ReportViewer({ report }: Props) {
  if (!report) {
    return <p className="text-gray-400 text-sm italic">No report generated yet</p>;
  }
  return (
    <div className="bg-white rounded-lg shadow p-4">
      <h3 className="text-sm font-medium text-gray-700 mb-3">Inspection Report</h3>
      <div className="prose prose-sm max-w-none text-gray-600 whitespace-pre-wrap">
        {report}
      </div>
    </div>
  );
}
