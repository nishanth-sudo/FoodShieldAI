import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { useInspection } from "../context/InspectionContext";
import { useLanguage } from "../context/LanguageContext";

interface ImageUploaderProps {
  onFileSelected?: (file: File) => void;
}

export function ImageUploader({ onFileSelected }: ImageUploaderProps) {
  const { validateAndSelectFile } = useInspection();
  const { t } = useLanguage();

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles && acceptedFiles[0]) {
        const file = acceptedFiles[0];
        const isValid = validateAndSelectFile(file);
        if (isValid && onFileSelected) {
          onFileSelected(file);
        }
      }
    },
    [validateAndSelectFile, onFileSelected]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpeg", ".jpg", ".png", ".webp"] },
    maxFiles: 1,
  });

  return (
    <div
      {...getRootProps()}
      className={`border-2 border-dashed rounded-xl p-8 md:p-12 text-center cursor-pointer transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 ${
        isDragActive
          ? "border-primary-500 bg-primary-50/50 dark:bg-primary-950/20"
          : "border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 hover:border-primary-500 hover:bg-gray-50/50 dark:hover:bg-gray-700/50"
      }`}
      role="button"
      tabIndex={0}
      aria-label={t.selectImage}
    >
      <input {...getInputProps()} id="dropzone-input" />
      <div className="flex flex-col items-center gap-4">
        <div className="text-5xl md:text-6xl text-gray-300 dark:text-gray-500">📷</div>
        <div className="space-y-1">
          <p className="text-gray-600 dark:text-gray-300 font-semibold text-base md:text-lg">
            {isDragActive ? t.dropHereText : t.dragDropText}
          </p>
          <p className="text-xs text-gray-400 dark:text-gray-500">
            {t.fileSizeHint}
          </p>
        </div>
        <span className="inline-block bg-primary-600 text-white px-5 py-2.5 rounded-lg text-sm font-semibold hover:bg-primary-700 active:scale-95 transition shadow-sm">
          {t.chooseImage}
        </span>
      </div>
    </div>
  );
}
export default ImageUploader;
