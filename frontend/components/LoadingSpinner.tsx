export function LoadingSpinner({ text = "Loading..." }: { text?: string }) {
  return (
    <div
      className="flex flex-col items-center justify-center py-16 animate-fadeIn"
      role="status"
      aria-live="polite"
    >
      <div className="relative w-12 h-12">
        {/* Outer glowing ring */}
        <div className="absolute inset-0 rounded-full border-4 border-primary-200 dark:border-primary-900" />
        {/* Spinning indicator */}
        <div className="absolute inset-0 rounded-full border-4 border-primary-600 border-t-transparent animate-spin" />
      </div>
      <p className="mt-4 text-sm font-semibold text-gray-500 dark:text-gray-400">{text}</p>
    </div>
  );
}
export default LoadingSpinner;
