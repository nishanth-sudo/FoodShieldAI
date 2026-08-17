export const CONFIG = {
  API_URL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  MAX_FILE_SIZE: 5 * 1024 * 1024, // 5MB
  ACCEPTED_FILE_TYPES: ["image/jpeg", "image/png", "image/webp"],
  ACCEPTED_FILE_EXTENSIONS: [".jpg", ".jpeg", ".png", ".webp"],
  CONFIDENCE_THRESHOLD: 0.70, // Confidence score below 70% shows caution warning
  POLL_INTERVAL_MS: 2000, // How often to poll for analysis results
  POLL_MAX_ATTEMPTS: 45, // Give up after ~90s of polling
};
