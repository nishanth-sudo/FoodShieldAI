import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

export const ImageUploader = () => {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    // TODO: Upload image to backend API
    // TODO: Show upload progress
    // TODO: Navigate to results page
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpeg', '.jpg', '.png', '.webp'] },
    maxSize: 10 * 1024 * 1024, // 10MB
  });

  return (
    <div {...getRootProps()}>
      <input {...getInputProps()} />
      {isDragActive ? <p>Drop images here...</p> : <p>Drag & drop food images, or click to select</p>}
    </div>
  );
};
