// Survey module card: DASS-21 self-report screening result.
import type { SubscaleResult, SurveyScoreResponse } from "../types/api";
import { NotDiagnosisNote } from "./common";

// Calm single-hue ramp (light -> deep teal). No red for any band; each band is always labelled in text.
const BAND_FILL = ["#d9ece9", "#b3d6d1", "#86bcb5", "#5d9f97", "#3f827b"];
const BAND_TEXT = ["#245550", "#245550", "#1b403c", "#ffffff", "#ffffff"];

function BandScale({ s, max }: { s: SubscaleResult; max: number }) {
  const span = max + 1;
  const marker = ((s.score + 0.5) / span) * 100;
  return (
    <div>
      <div className="relative pt-7" aria-hidden>
        <div
          className="absolute top-0 -translate-x-1/2 whitespace-nowrap rounded bg-ink px-1.5 py-0.5 text-xs font-semibold text-white"
          style={{ left: `clamp(16px, ${marker}%, calc(100% - 16px))` }}
        >
          {s.score}
        </div>
        <div className="absolute top-5 h-2 w-0.5 -translate-x-1/2 bg-ink" style={{ left: `${marker}%` }} />
        <div className="flex h-7 overflow-hidden rounded-md">
          {s.bands.map((b, i) => (
            <div
              key={b.band}
              className="flex items-center justify-center border-r-2 border-card text-[11px] font-medium last:border-r-0"
              style={{ width: `${((b.max - b.min + 1) / span) * 100}%`, background: BAND_FILL[i], color: BAND_TEXT[i] }}
              title={`${b.band}: ${b.min}–${b.max}`}
            >
              {/* In-bar labels only where they fit; the full band list is always printed below. */}
              {(b.max - b.min + 1) / span >= 0.2 + b.band.length * 0.008 && (
                <span className="hidden whitespace-nowrap px-1 md:inline">{b.band}</span>
              )}
            </div>
          ))}
        </div>
        <div className="mt-1 flex justify-between text-xs text-ink-muted">
          <span>0</span>
          <span>{max}</span>
        </div>
      </div>
      <ul className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs text-ink-soft">
        {s.bands.map((b, i) => (
          <li key={b.band} className={i === s.band_index ? "font-semibold text-ink" : undefined}>
            {b.band} {b.min}–{b.max}
            {i === s.band_index && <span className="sr-only"> (your score)</span>}
          </li>
        ))}
      </ul>
    </div>
  );
}

function summary(r: SurveyScoreResponse): string {
  if (r.tied_states.length > 1) {
    const band = r.subscales.find((s) => s.subscale === r.tied_states[0])?.band;
    const names = r.tied_states.join(" and ").replace(/ and (?=[^ ]+ and )/, ", ");
    return `${names} are tied in the ${band} band, so no single scale stands out.`;
  }
  if (r.dominant_state === "Normal") return "All three scores are in the Normal range.";
  const top = r.subscales.find((s) => s.subscale === r.dominant_state);
  return `Your most elevated scale is ${r.dominant_state}, in the ${top?.band} band.`;
}

export function SurveyResultCard({ result }: { result: SurveyScoreResponse }) {
  const tied = result.tied_states.length > 1;
  return (
    <article className="card" aria-labelledby="survey-card-title" data-testid="survey-card">
      <p className="text-sm font-medium uppercase tracking-wide text-teal-700">Wellbeing questionnaire</p>
      <h2 id="survey-card-title" className="h2 mt-1">
        DASS-21 self-report screening result
      </h2>
      <p className="mt-1 text-ink-soft">Based only on your own answers about the past week. Each scale runs from 0 to {result.max_score}.</p>

      <p className="mt-4 rounded-lg bg-teal-50 px-4 py-3 font-medium text-teal-800" data-testid={tied ? "survey-tie" : "survey-summary"}>
        {summary(result)}
      </p>

      <div className="mt-6 space-y-7">
        {result.subscales.map((s) => (
          <section key={s.subscale} aria-label={`${s.subscale} scale`} data-testid={`row-${s.subscale.toLowerCase()}`}>
            <div className="mb-1 flex flex-wrap items-baseline justify-between gap-2">
              <h3 className="text-lg font-semibold">{s.subscale}</h3>
              <p className="text-ink-soft">
                Score <span className="font-semibold text-ink">{s.score}</span> · Band{" "}
                <span className="font-semibold text-ink">{s.band}</span>
              </p>
            </div>
            <BandScale s={s} max={result.max_score} />
          </section>
        ))}
      </div>

      {result.supportive_message && (
        <p className="mt-6 rounded-xl border border-teal-200 bg-teal-50 px-4 py-3 text-teal-800" data-testid="supportive-message">
          {result.supportive_message}
        </p>
      )}

      <p className="mt-5 text-sm text-ink-muted">{result.notice}</p>
      <NotDiagnosisNote />
    </article>
  );
}
