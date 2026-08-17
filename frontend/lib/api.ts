import axios from "axios";
import { getSession } from "next-auth/react";
import { CONFIG } from "./config";
import type {
  HealthStatus,
  Inspection,
  PaginatedInspections,
  UploadResponse,
  User,
} from "./types";

export const api = axios.create({
  baseURL: CONFIG.API_URL,
  headers: { "Content-Type": "application/json" },
});

api.interceptors.request.use(async (config) => {
  if (typeof window !== "undefined") {
    const session = await getSession();
    if (session?.accessToken) {
      config.headers.Authorization = `Bearer ${session.accessToken}`;
    }
  }
  return config;
});

api.interceptors.response.use(
  (res) => res,
  async (error) => {
    if (error.response?.status === 401 && typeof window !== "undefined") {
      const session = await getSession();
      if (session?.refreshToken && !error.config?._retry) {
        error.config._retry = true;
        try {
          const { data } = await axios.post<{ access_token: string }>(
            `${CONFIG.API_URL}/auth/refresh`,
            { refresh_token: session.refreshToken }
          );
          error.config.headers.Authorization = `Bearer ${data.access_token}`;
          return api(error.config);
        } catch {
          window.location.href = "/login";
        }
      }
    }
    return Promise.reject(error);
  }
);

export const authApi = {
  register: (data: { email: string; password: string; name: string }) =>
    api.post("/auth/register", data),
  login: (data: { email: string; password: string }) =>
    api.post<{ access_token: string; refresh_token: string }>("/auth/login", data),
  me: () => api.get<User>("/auth/me"),
};

export const inspectionApi = {
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return api.post<UploadResponse>("/inspections/upload", form, {
      headers: { "Content-Type": "multipart/form-data" },
    });
  },
  get: (id: string) => api.get<Inspection>(`/inspections/${id}`),
  list: (page = 1, limit = 20) =>
    api.get<PaginatedInspections>(`/inspections?page=${page}&limit=${limit}`),
};

export const adminApi = {
  users: () => api.get<User[]>("/admin/users"),
  inspections: () => api.get<Inspection[]>("/admin/inspections"),
  health: () => api.get<HealthStatus>("/health"),
};
