// In-memory session state only: no localStorage, no history.
// Cleared when the user leaves the Results page (see RouteWatcher in App.tsx).
import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from "react";
import type { ModelInfoResponse, PredictionResponse, SurveyScoreResponse } from "../types/api";
import { getModelInfo } from "../services/api";

export interface VoiceSession {
  result: PredictionResponse;
  audioUrl: string; // object URL for playback, revoked on clear
}

interface SessionValue {
  voice: VoiceSession | null;
  survey: SurveyScoreResponse | null;
  setVoice: (result: PredictionResponse, audio: Blob) => void;
  setSurvey: (result: SurveyScoreResponse) => void;
  clearAll: () => void;
  modelInfo: ModelInfoResponse | null;
  modelInfoError: unknown;
  loadModelInfo: () => Promise<ModelInfoResponse>;
}

const SessionContext = createContext<SessionValue | null>(null);

export function SessionProvider({ children }: { children: ReactNode }) {
  const [voice, setVoiceState] = useState<VoiceSession | null>(null);
  const [survey, setSurveyState] = useState<SurveyScoreResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [modelInfoError, setModelInfoError] = useState<unknown>(null);
  const pending = useRef<Promise<ModelInfoResponse> | null>(null);

  const setVoice = useCallback((result: PredictionResponse, audio: Blob) => {
    setVoiceState((prev) => {
      if (prev) URL.revokeObjectURL(prev.audioUrl);
      return { result, audioUrl: URL.createObjectURL(audio) };
    });
  }, []);

  const setSurvey = useCallback((result: SurveyScoreResponse) => setSurveyState(result), []);

  const clearAll = useCallback(() => {
    setVoiceState((prev) => {
      if (prev) URL.revokeObjectURL(prev.audioUrl);
      return null;
    });
    setSurveyState(null);
  }, []);

  const loadModelInfo = useCallback(() => {
    if (modelInfo) return Promise.resolve(modelInfo);
    if (!pending.current) {
      pending.current = getModelInfo()
        .then((info) => {
          setModelInfo(info);
          setModelInfoError(null);
          return info;
        })
        .catch((e) => {
          setModelInfoError(e);
          pending.current = null;
          throw e;
        });
    }
    return pending.current;
  }, [modelInfo]);

  const value = useMemo(
    () => ({ voice, survey, setVoice, setSurvey, clearAll, modelInfo, modelInfoError, loadModelInfo }),
    [voice, survey, setVoice, setSurvey, clearAll, modelInfo, modelInfoError, loadModelInfo],
  );
  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession(): SessionValue {
  const ctx = useContext(SessionContext);
  if (!ctx) throw new Error("useSession must be used inside SessionProvider");
  return ctx;
}
