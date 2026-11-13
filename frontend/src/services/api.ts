// Every backend call goes through this module.
import type {
  ErrorResponse,
  HealthResponse,
  ModelInfoResponse,
  PredictionResponse,
  SurveyQuestionsResponse,
  SurveyScoreRequest,
  SurveyScoreResponse,
} from "../types/api";

const RAW_BASE = import.meta.env.VITE_API_URL;
if (!RAW_BASE) {
  // Fail loudly in development rather than calling the wrong host.
  console.error("VITE_API_URL is not set. Copy frontend/.env.example to .env.local and set it.");
}
export const API_BASE = (RAW_BASE ?? "").replace(/\/+$/, "");

export class ApiError extends Error {
  readonly errorCode: string;
  readonly status: number;
  constructor(errorCode: string, message: string, status: number) {
    super(message);
    this.errorCode = errorCode;
    this.status = status;
  }
}

function toApiError(status: number, body: unknown): ApiError {
  const b = body as Partial<ErrorResponse> | null;
  if (b && typeof b.error_code === "string") {
    return new ApiError(b.error_code, b.message ?? "", status);
  }
  return new ApiError("unknown_error", `Unexpected response (${status})`, status);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new ApiError("network_error", "Could not reach the server.", 0);
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) throw toApiError(res.status, body);
  return body as T;
}

export const getHealth = () => request<HealthResponse>("/api/health");

export const getModelInfo = () => request<ModelInfoResponse>("/api/model-info");

export const getSurveyQuestions = () => request<SurveyQuestionsResponse>("/api/survey/questions");

export const scoreSurvey = (payload: SurveyScoreRequest) =>
  request<SurveyScoreResponse>("/api/survey/score", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

/**
 * Upload a clip for analysis. Uses XHR so real upload progress can be shown.
 * `onUploadProgress` receives 0..1; it reaches 1 when the server has the file.
 */
export function analyzeAudio(
  audio: Blob,
  filename: string,
  onUploadProgress?: (fraction: number) => void,
  signal?: AbortSignal,
): Promise<PredictionResponse> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("audio", audio, filename);
    form.append("consent", "true");

    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/api/analyze`);
    xhr.responseType = "json";
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onUploadProgress?.(e.loaded / e.total);
    };
    xhr.upload.onload = () => onUploadProgress?.(1);
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(xhr.response as PredictionResponse);
      else reject(toApiError(xhr.status, xhr.response));
    };
    xhr.onerror = () => reject(new ApiError("network_error", "Could not reach the server.", 0));
    xhr.onabort = () => reject(new ApiError("aborted", "Upload cancelled.", 0));
    signal?.addEventListener("abort", () => xhr.abort());
    xhr.send(form);
  });
}
