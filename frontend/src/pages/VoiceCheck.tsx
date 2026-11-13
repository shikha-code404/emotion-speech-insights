import { useCallback, useEffect, useRef, useState } from "react";
import { analyzeAudio } from "../services/api";
import { useSession } from "../state/session";
import { ConsentCheckbox, ErrorNotice, NextStepPanel } from "../components/common";
import { ProgressSteps } from "../components/ProgressSteps";
import { Waveform, formatSeconds, useDecodedAudio } from "../components/Waveform";

const MIN_S = 5;
const MAX_S = 15;
const MIME_CANDIDATES = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];

const STEPS = [
  { label: "Uploading your clip" },
  { label: "Analysing on the server", detail: "Checking the audio, measuring vocal features and estimating the emotion." },
  { label: "Preparing your result" },
];

type Mode = "choose" | "record" | "upload";
type RecState = "idle" | "requesting" | "recording";

interface Clip {
  blob: Blob;
  filename: string;
  url: string;
  source: "recording" | "upload";
}

function extFor(mime: string): string {
  if (mime.includes("webm")) return "webm";
  if (mime.includes("ogg")) return "ogg";
  if (mime.includes("mp4")) return "m4a";
  return "webm";
}

export default function VoiceCheck() {
  const { setVoice, voice, survey } = useSession();
  const [consent, setConsent] = useState(false);
  const [mode, setMode] = useState<Mode>("choose");
  const [recState, setRecState] = useState<RecState>("idle");
  const [elapsed, setElapsed] = useState(0);
  const [micError, setMicError] = useState<string | null>(null);
  const [clip, setClip] = useState<Clip | null>(null);
  const [step, setStep] = useState(-1);
  const [uploadPct, setUploadPct] = useState(0);
  const [error, setError] = useState<unknown>(null);
  const [done, setDone] = useState(false);

  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const startRef = useRef(0);
  const discardRef = useRef(false);
  const clipRef = useRef<Clip | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  // Keyboard focus follows the flow: when the control the user pressed disappears,
  // focus moves to the next control instead of falling back to the top of the page.
  const cancelRef = useRef<HTMLButtonElement>(null);
  const analyseRef = useRef<HTMLButtonElement>(null);
  const firstChoiceRef = useRef<HTMLButtonElement>(null);
  const allowRef = useRef<HTMLButtonElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const decoded = useDecodedAudio(clip?.blob ?? null);

  useEffect(() => {
    if (recState === "recording") cancelRef.current?.focus();
  }, [recState]);
  useEffect(() => {
    if (clip) analyseRef.current?.focus();
  }, [clip]);
  useEffect(() => {
    if (!consent || clip) return;
    if (mode === "choose") firstChoiceRef.current?.focus({ preventScroll: true });
    if (mode === "record") allowRef.current?.focus();
    if (mode === "upload") fileRef.current?.focus();
  }, [mode]); // eslint-disable-line react-hooks/exhaustive-deps

  const releaseMic = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (timerRef.current) window.clearInterval(timerRef.current);
    timerRef.current = null;
  }, []);

  const replaceClip = useCallback((next: Clip | null) => {
    const prev = clipRef.current;
    if (prev) URL.revokeObjectURL(prev.url);
    clipRef.current = next;
    setClip(next);
  }, []);

  // On leaving the page: stop the microphone, abort uploads, drop the clip.
  useEffect(
    () => () => {
      discardRef.current = true;
      if (recorderRef.current?.state === "recording") recorderRef.current.stop();
      releaseMic();
      abortRef.current?.abort();
      if (clipRef.current) URL.revokeObjectURL(clipRef.current.url);
    },
    [releaseMic],
  );

  const stopRecording = useCallback(() => {
    if (recorderRef.current?.state === "recording") recorderRef.current.stop();
    releaseMic(); // stops every track, so the browser's microphone indicator turns off
  }, [releaseMic]);

  const startRecording = async () => {
    setMicError(null);
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setMicError("This browser can't record audio. Please upload a file instead.");
      return;
    }
    setRecState("requesting");
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
      setRecState("idle");
      const name = (e as DOMException)?.name;
      setMicError(
        name === "NotAllowedError"
          ? "Microphone permission was declined. You can allow it in your browser's site settings, or upload a file instead."
          : name === "NotFoundError"
            ? "No microphone was found. Please connect one, or upload a file instead."
            : "The microphone couldn't be started. Please try again or upload a file instead.",
      );
      return;
    }
    streamRef.current = stream;
    const mimeType = MIME_CANDIDATES.find((m) => MediaRecorder.isTypeSupported(m)) ?? "";
    const rec = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    recorderRef.current = rec;
    chunksRef.current = [];
    discardRef.current = false;
    rec.ondataavailable = (e) => e.data.size > 0 && chunksRef.current.push(e.data);
    rec.onstop = () => {
      setRecState("idle");
      if (discardRef.current) return;
      const type = rec.mimeType || mimeType || "audio/webm";
      const blob = new Blob(chunksRef.current, { type });
      chunksRef.current = [];
      replaceClip({ blob, filename: `recording.${extFor(type)}`, url: URL.createObjectURL(blob), source: "recording" });
    };
    rec.start(250);
    startRef.current = performance.now();
    setElapsed(0);
    setRecState("recording");
    timerRef.current = window.setInterval(() => {
      const s = (performance.now() - startRef.current) / 1000;
      setElapsed(s);
      if (s >= MAX_S) stopRecording();
    }, 100);
  };

  const cancelRecording = () => {
    discardRef.current = true;
    stopRecording();
  };

  const onFile = (file: File | undefined) => {
    setError(null);
    if (!file) return;
    replaceClip({ blob: file, filename: file.name, url: URL.createObjectURL(file), source: "upload" });
  };

  const deleteClip = () => {
    replaceClip(null);
    setError(null);
  };

  const analyse = async () => {
    if (!clip) return;
    setError(null);
    setStep(0);
    setUploadPct(0);
    abortRef.current = new AbortController();
    try {
      const result = await analyzeAudio(
        clip.blob,
        clip.filename,
        (f) => {
          setUploadPct(Math.round(f * 100));
          if (f >= 1) setStep(1);
        },
        abortRef.current.signal,
      );
      setStep(2);
      setVoice(result, clip.blob);
      // The session made its own object URL for playback; drop the page copy.
      replaceClip(null);
      setStep(3);
      setDone(true);
    } catch (e) {
      setError(e);
      setStep(-1);
    }
  };

  const busy = step >= 0 && step < 3;
  const seconds = Math.floor(elapsed);

  if (done && voice) {
    return (
      <div className="space-y-6">
        <h1 className="h1">Voice check</h1>
        <NextStepPanel
          doneLabel="Voice check complete"
          otherDone={!!survey}
          otherTo="/questionnaire"
          otherLabel="Also take the wellbeing questionnaire"
        />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="h1">Voice check</h1>
        <p className="mt-2 text-ink-soft">
          Speak naturally for 5–15 seconds about anything, for example how your day is going. The model estimates the
          emotion your voice expresses in this clip only.
        </p>
      </div>

      <ConsentCheckbox id="voice-consent" checked={consent} onChange={setConsent}>
        I agree to send this clip to the server for analysis. I understand it is deleted right after analysis, that the
        result describes vocal expression in this clip only, and that it is not a diagnosis.
      </ConsentCheckbox>

      {consent && (
        <section className="card space-y-5" aria-label="Your clip">
          {!clip && mode === "choose" && (
            <div className="flex flex-col gap-3 sm:flex-row">
              <button ref={firstChoiceRef} type="button" className="btn-primary" onClick={() => setMode("record")}>
                Record with microphone
              </button>
              <button type="button" className="btn-secondary" onClick={() => setMode("upload")}>
                Upload an audio file
              </button>
            </div>
          )}

          {!clip && mode === "record" && (
            <div className="space-y-4">
              {recState !== "recording" ? (
                <>
                  <p className="text-ink-soft">
                    Your browser will ask for microphone permission. Recording stops automatically at {MAX_S} seconds,
                    and the microphone is switched off as soon as you stop.
                  </p>
                  <div className="flex flex-col gap-3 sm:flex-row">
                    <button ref={allowRef} type="button" className="btn-primary" onClick={startRecording} disabled={recState === "requesting"}>
                      {recState === "requesting" ? "Waiting for permission…" : "Allow microphone and start recording"}
                    </button>
                    <button type="button" className="btn-secondary" onClick={() => setMode("choose")}>
                      Back
                    </button>
                  </div>
                </>
              ) : (
                <div className="space-y-4">
                  <p className="flex items-center gap-2 font-medium text-teal-800" aria-live="polite">
                    <span className="h-3 w-3 animate-pulse rounded-full bg-teal-600" aria-hidden />
                    Recording… <span data-testid="rec-elapsed">{seconds}</span> s of {MAX_S}
                  </p>
                  <div className="h-2 overflow-hidden rounded-full bg-sand-200" aria-hidden>
                    <div className="h-full bg-teal-500 transition-all" style={{ width: `${Math.min(100, (elapsed / MAX_S) * 100)}%` }} />
                  </div>
                  <div className="flex flex-col gap-3 sm:flex-row">
                    <button type="button" className="btn-primary" onClick={stopRecording} disabled={elapsed < MIN_S}>
                      {elapsed < MIN_S ? `Stop (available after ${MIN_S} s)` : "Stop recording"}
                    </button>
                    <button ref={cancelRef} type="button" className="btn-secondary" onClick={cancelRecording}>
                      Cancel
                    </button>
                  </div>
                </div>
              )}
              {micError && (
                <p role="alert" className="rounded-lg bg-sand-100 px-3 py-2 text-ink-soft">
                  {micError}
                </p>
              )}
            </div>
          )}

          {!clip && mode === "upload" && (
            <div className="space-y-3">
              <label htmlFor="audio-file" className="block font-medium">
                Choose an audio file (WAV, MP3, WebM, OGG or M4A, 1–30 seconds)
              </label>
              <input
                ref={fileRef}
                id="audio-file"
                type="file"
                accept="audio/*,.wav,.mp3,.webm,.ogg,.m4a"
                className="block w-full text-ink-soft file:mr-4 file:rounded-lg file:border-0 file:bg-teal-600 file:px-4 file:py-2 file:text-white"
                onChange={(e) => onFile(e.target.files?.[0])}
              />
              <button type="button" className="btn-secondary" onClick={() => setMode("choose")}>
                Back
              </button>
            </div>
          )}

          {clip && (
            <div className="space-y-4" data-testid="clip-preview">
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <h2 className="h2">{clip.source === "recording" ? "Your recording" : clip.filename}</h2>
                <p className="text-ink-soft">
                  Duration: <span data-testid="clip-duration">{formatSeconds(decoded.duration)}</span>
                </p>
              </div>
              {decoded.peaks ? (
                <Waveform peaks={decoded.peaks} />
              ) : decoded.failed ? (
                <p className="text-sm text-ink-muted">A waveform preview isn't available for this file type, but it can still be analysed.</p>
              ) : (
                <div className="h-20 animate-pulse rounded bg-sand-100" aria-hidden />
              )}
              <audio controls src={clip.url} className="w-full" aria-label="Play back your clip" />
              <div className="flex flex-col gap-3 sm:flex-row">
                <button ref={analyseRef} type="button" className="btn-primary" onClick={analyse} disabled={busy}>
                  Analyse this clip
                </button>
                {clip.source === "recording" ? (
                  <button type="button" className="btn-secondary" disabled={busy} onClick={() => { deleteClip(); setMode("record"); }}>
                    Re-record
                  </button>
                ) : (
                  <button type="button" className="btn-secondary" disabled={busy} onClick={() => { deleteClip(); setMode("upload"); }}>
                    Choose a different file
                  </button>
                )}
                <button type="button" className="btn-ghost" disabled={busy} onClick={() => { deleteClip(); setMode("choose"); }}>
                  Delete clip
                </button>
              </div>
            </div>
          )}

          {step >= 0 && (
            <div className="border-t border-line pt-4" aria-live="polite">
              <ProgressSteps
                steps={STEPS.map((s, i) => (i === 0 && step === 0 ? { ...s, detail: `${uploadPct}% uploaded` } : s))}
                current={step}
              />
            </div>
          )}

          {error != null && <ErrorNotice error={error} />}
        </section>
      )}
    </div>
  );
}
