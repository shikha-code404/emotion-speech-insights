// Mirrors backend/app/schemas.py. Keep the two in sync.

// ---------------------------------------------------------------------------
// Speech module
// ---------------------------------------------------------------------------

export interface EmotionScore {
  emotion: string;
  score: number; // 0..1
}

export type Uncertainty = "low" | "moderate" | "high";

export interface PredictionResponse {
  prediction_type: "vocal_emotion" | string;
  predicted_emotion: string;
  confidence: number; // 0..1
  scores: EmotionScore[];
  uncertainty: Uncertainty;
  model_version: string;
  audio_duration_seconds: number;
  notice: string;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  survey_loaded: boolean;
}

export interface SurveyModuleInfo {
  version: string;
  instrument: string;
  development_data_synthetic: boolean;
  limitations: string[];
}

export interface ModelInfoResponse {
  version: string;
  model_family: string;
  labels: string[];
  corpora: string[];
  metrics: Record<string, unknown>;
  limitations: string[];
  survey: SurveyModuleInfo;
}

// ---------------------------------------------------------------------------
// Survey module (DASS-21 self-report screening)
// ---------------------------------------------------------------------------

export interface AnswerOption {
  value: number;
  label: string;
}

export interface SurveyItem {
  id: string; // "q1".."q21"
  number: number;
  text: string;
}

export interface SurveyQuestionsResponse {
  version: string;
  instructions: string;
  answer_scale: AnswerOption[];
  items: SurveyItem[];
  notice: string;
}

export interface SurveyScoreRequest {
  consent: boolean;
  answers: Record<string, number>;
}

export interface BandRange {
  band: string;
  min: number;
  max: number;
}

export interface SubscaleResult {
  subscale: string;
  score: number; // 0..42
  band: string;
  band_index: number; // 0..4
  bands: BandRange[];
}

export interface SurveyScoreResponse {
  result_type: "dass21_self_report_screening" | string;
  subscales: SubscaleResult[];
  max_score: number;
  dominant_state: string;
  tied_states: string[];
  supportive_message: string | null;
  notice: string;
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

export interface ErrorResponse {
  error_code: string;
  message: string;
}
