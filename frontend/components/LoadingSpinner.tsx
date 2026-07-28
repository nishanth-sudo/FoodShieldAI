export function LoadingSpinner({ text = "Loading..." }: { text?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-12">
      <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-primary-600" />
      <p className="mt-3 text-sm text-gray-500">{text}</p>
    </div>
  );
}
