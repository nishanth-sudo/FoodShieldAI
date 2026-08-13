import React, { createContext, useContext, useState, useEffect, useCallback } from "react";
import { CONFIG } from "../lib/config";
import { inspectionApi } from "../lib/api";

export type InspectionStatus = "idle" | "validating" | "uploading" | "analyzing" | "completed" | "failed";

interface InspectionContextType {
  file: File | null;
  preview: string | null;
  status: InspectionStatus;
  statusText: string;
  result: any | null;
  error: string | null;
  isOnline: boolean;
  localHistory: any[];
  validateAndSelectFile: (selectedFile: File) => boolean;
  clearFile: () => void;
  runInspection: () => Promise<void>;
  exportInspectionAsJson: (inspection: any) => void;
  deleteLocalInspection: (id: string) => void;
}

const InspectionContext = createContext<InspectionContextType>({} as InspectionContextType);

export function InspectionProvider({ children }: { children: React.ReactNode }) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [status, setStatus] = useState<InspectionStatus>("idle");
  const [statusText, setStatusText] = useState<string>("");
  const [result, setResult] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isOnline, setIsOnline] = useState<boolean>(true);
  const [localHistory, setLocalHistory] = useState<any[]>([]);

  // Monitor network status
  useEffect(() => {
    if (typeof window !== "undefined") {
      setIsOnline(navigator.onLine);
      const handleOnline = () => setIsOnline(true);
      const handleOffline = () => setIsOnline(false);

      window.addEventListener("online", handleOnline);
      window.addEventListener("offline", handleOffline);

      // Load local history from localStorage
      const cachedHistory = localStorage.getItem("local_inspections");
      if (cachedHistory) {
        try {
          setLocalHistory(JSON.parse(cachedHistory));
        } catch (e) {
          console.error("Failed to parse cached history", e);
        }
      }

      return () => {
        window.removeEventListener("online", handleOnline);
        window.removeEventListener("offline", handleOffline);
      };
    }
  }, []);

  // Update offline warning automatically
  useEffect(() => {
    if (!isOnline) {
      setError("offlineWarning");
    } else if (error === "offlineWarning") {
      setError(null);
    }
  }, [isOnline, error]);

  const validateAndSelectFile = useCallback((selectedFile: File): boolean => {
    setError(null);
    setResult(null);

    // Validate type
    if (!CONFIG.ACCEPTED_FILE_TYPES.includes(selectedFile.type)) {
      setError("invalidFileType");
      setStatus("failed");
      return false;
    }

    // Validate size
    if (selectedFile.size > CONFIG.MAX_FILE_SIZE) {
      setError("fileTooLarge");
      setStatus("failed");
      return false;
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setStatus("idle");
    setStatusText("");
    return true;
  }, []);

  const clearFile = useCallback(() => {
    setFile(null);
    if (preview) {
      URL.revokeObjectURL(preview);
    }
    setPreview(null);
    setStatus("idle");
    setStatusText("");
    setResult(null);
    setError(null);
  }, [preview]);

  const saveInspectionLocally = useCallback((inspection: any) => {
    setLocalHistory((prev) => {
      // Avoid duplicate keys
      const filtered = prev.filter((item) => item.id !== inspection.id);
      const updated = [inspection, ...filtered];
      localStorage.setItem("local_inspections", JSON.stringify(updated));
      return updated;
    });
  }, []);

  const deleteLocalInspection = useCallback((id: string) => {
    setLocalHistory((prev) => {
      const updated = prev.filter((item) => item.id !== id);
      localStorage.setItem("local_inspections", JSON.stringify(updated));
      return updated;
    });
  }, []);

  const exportInspectionAsJson = useCallback((inspection: any) => {
    if (!inspection) return;
    const jsonString = `data:text/json;charset=utf-8,${encodeURIComponent(
      JSON.stringify(inspection, null, 2)
    )}`;
    const downloadAnchor = document.createElement("a");
    downloadAnchor.setAttribute("href", jsonString);
    downloadAnchor.setAttribute("download", `foodshield_inspection_${inspection.id || "export"}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  }, []);

  const runInspection = useCallback(async () => {
    if (!file) return;

    if (!isOnline) {
      setError("offlineWarning");
      return;
    }

    setStatus("uploading");
    setStatusText("Uploading image...");
    setError(null);

    try {
      const { data } = await inspectionApi.upload(file);
      setStatus("analyzing");
      setStatusText("Analyzing food quality...");

      let attempts = 0;
      const pollInterval = setInterval(async () => {
        try {
          const { data: updated } = await inspectionApi.get(data.id);
          attempts += 1;

          if (updated.status === "completed") {
            clearInterval(pollInterval);
            setResult(updated);
            saveInspectionLocally(updated);
            setStatus("completed");
            setStatusText("");
          } else if (updated.status === "failed") {
            clearInterval(pollInterval);
            setError(updated.error || "Inspection analysis failed on the model server.");
            setStatus("failed");
            setStatusText("");
          } else {
            // still processing
            setStatusText(`Analyzing food quality... (polling attempt ${attempts})`);
          }
        } catch (pollErr: any) {
          clearInterval(pollInterval);
          setError(pollErr.response?.data?.detail || "Error fetching inspection results.");
          setStatus("failed");
          setStatusText("");
        }
      }, CONFIG.RETRY_INITIAL_DELAY * 2);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || "Failed to upload image. Please try again.");
      setStatus("failed");
      setStatusText("");
    }
  }, [file, isOnline, saveInspectionLocally]);

  return (
    <InspectionContext.Provider
      value={{
        file,
        preview,
        status,
        statusText,
        result,
        error,
        isOnline,
        localHistory,
        validateAndSelectFile,
        clearFile,
        runInspection,
        exportInspectionAsJson,
        deleteLocalInspection,
      }}
    >
      {children}
    </InspectionContext.Provider>
  );
}

export const useInspection = () => useContext(InspectionContext);
