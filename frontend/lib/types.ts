export interface User {
  id: string;
  email: string;
  name: string;
  role: string;
}

export interface PackagingDefect {
  type: string;
  confidence: number;
}

/** Matches backend/domain/schemas.py InspectionResponse */
export interface Inspection {
  id: string;
  status: string;
  image_url: string;
  image_thumbnail_url?: string | null;
  food_type?: string | null;
  freshness_score?: number | null;
  shelf_life_days?: number | null;
  packaging_defects?: PackagingDefect[] | null;
  contamination_risks?: Record<string, number> | null;
  ocr_data?: Record<string, unknown> | null;
  xai_heatmap_url?: string | null;
  confidence_scores?: Record<string, number> | null;
  report?: string | null;
  created_at?: string | null;
  completed_at?: string | null;
  error?: string | null;
}

export interface PaginatedInspections {
  items: Inspection[];
  total: number;
  page: number;
  limit: number;
}

export interface UploadResponse {
  id: string;
}

export interface HealthStatus {
  status: string;
  database: string;
  service: string;
}

/**
 * Extract the overall classification confidence from the backend's
 * `confidence_scores` dict (e.g. { food_classification: 0.92, ... }).
 * Falls back to the highest score if the key is missing.
 */
export function getConfidence(inspection: Inspection): number | undefined {
  const scores = inspection.confidence_scores;
  if (!scores) return undefined;
  const values = Object.values(scores).filter((v): v is number => typeof v === "number");
  if (values.length === 0) return undefined;
  return Math.max(...values);
}
