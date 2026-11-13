import { ApiError } from "../services/api";

// One friendly message per backend error_code (backend/app/errors.py),
// plus client-side codes (network_error, unknown_error).
const MESSAGES: Record<string, { title: string; hint: string }> = {
  consent_required: {
    title: "Consent is needed first",
    hint: "Please tick the consent box, then try again.",
  },
  file_too_large: {
    title: "That file is too large",
    hint: "Please use a shorter clip (under 10 MB).",
  },
  unsupported_format: {
    title: "That file type isn't supported",
    hint: "Please use a WAV, MP3, WebM, OGG or M4A audio file.",
  },
  audio_too_short: {
    title: "The clip is too short",
    hint: "Please record or upload at least a few seconds of speech.",
  },
  audio_too_long: {
    title: "The clip is too long",
    hint: "Please keep it under 30 seconds. Recordings here stop at 15 seconds.",
  },
  audio_silent: {
    title: "We couldn't hear any speech",
    hint: "Check that the right microphone is selected and not muted, then try again.",
  },
  decode_failed: {
    title: "We couldn't read that audio",
    hint: "The file may be damaged or in an unusual format. Try recording here instead, or export it as WAV.",
  },
  model_not_loaded: {
    title: "Voice analysis isn't available right now",
    hint: "The service is starting up or under maintenance. Please try again in a moment.",
  },
  survey_not_loaded: {
    title: "The questionnaire isn't available right now",
    hint: "Please try again in a moment.",
  },
  invalid_answers: {
    title: "Some answers are missing",
    hint: "Please go back and answer every question.",
  },
  invalid_request: {
    title: "Something in the request wasn't right",
    hint: "Please try again. If it keeps happening, reload the page.",
  },
  network_error: {
    title: "We couldn't reach the server",
    hint: "Check your internet connection and try again.",
  },
  unknown_error: {
    title: "Something went wrong",
    hint: "Please try again in a moment.",
  },
};

export function friendlyError(err: unknown): { code: string; title: string; hint: string } {
  const code = err instanceof ApiError ? err.errorCode : "unknown_error";
  const m = MESSAGES[code] ?? MESSAGES.unknown_error;
  return { code, ...m };
}

export function capitalise(s: string): string {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s;
}
