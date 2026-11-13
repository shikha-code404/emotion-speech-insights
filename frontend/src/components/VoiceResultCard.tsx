// Speech module card. By design this card never uses clinical or survey wording.
import { useState } from "react";
import { Bar, BarChart, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { PredictionResponse, Uncertainty } from "../types/api";
import { capitalise } from "../lib/errors";
import { NotDiagnosisNote } from "./common";

const CONFIDENCE_WORDS: Record<Uncertainty, string> = {
  low: "The model is fairly confident about this.",
  moderate: "The model is only somewhat sure. Other expressions were also likely.",
  high: "The model is unsure. Several expressions scored about the same.",
};

const REFLECTION_REPLIES: Record<string, string> = {
  yes: "Thanks for checking in with yourself. Noticing how you sound can be a useful moment of awareness.",
  partly: "That's common. A few seconds of voice only captures part of how someone feels.",
  no: "That's completely fine. You know how you feel far better than a model listening to a short clip.",
};

export function VoiceResultCard({
  result,
  audioUrl,
  labels,
}: {
  result: PredictionResponse;
  audioUrl: string;
  labels: string[] | null; // display order from /api/model-info
}) {
  const [reflection, setReflection] = useState<string | null>(null);
  const byLabel = new Map(result.scores.map((s) => [s.emotion, s.score]));
  const order = labels?.length ? labels : result.scores.map((s) => s.emotion);
  const data = order
    .filter((l) => byLabel.has(l))
    .map((l) => ({ label: capitalise(l), key: l, pct: Math.round((byLabel.get(l) ?? 0) * 100) }));
  const ranked = [...result.scores].sort((a, b) => b.score - a.score);
  const uncertain = result.uncertainty === "high";
  const pct = Math.round(result.confidence * 100);

  return (
    <article className="card" aria-labelledby="voice-card-title" data-testid="voice-card">
      <p className="text-sm font-medium uppercase tracking-wide text-teal-700">Voice check</p>
      <h2 id="voice-card-title" className="h2 mt-1">
        {uncertain ? (
          <>The model couldn't tell clearly from this clip</>
        ) : (
          <>
            Most likely vocal expression: <span className="text-teal-700">{capitalise(result.predicted_emotion)}</span>
          </>
        )}
      </h2>
      {uncertain && (
        <p className="mt-2 text-ink-soft" data-testid="voice-uncertain">
          Closest matches: {capitalise(ranked[0].emotion)} and {capitalise(ranked[1].emotion)}. Try a slightly longer
          clip in a quiet place if you'd like a clearer estimate.
        </p>
      )}
      <p className="mt-2 text-ink-soft" data-testid="voice-confidence">
        {CONFIDENCE_WORDS[result.uncertainty]} ({pct}% for {capitalise(result.predicted_emotion)})
      </p>

      <figure className="mt-5">
        <figcaption className="mb-2 text-sm font-medium text-ink-soft">Score for each expression the model knows</figcaption>
        <div style={{ height: Math.max(180, data.length * 40) }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} layout="vertical" margin={{ top: 0, right: 48, bottom: 0, left: 0 }} barCategoryGap={8}>
              <XAxis type="number" domain={[0, 100]} hide />
              <YAxis type="category" dataKey="label" width={92} tickLine={false} axisLine={false} tick={{ fill: "#2b2a28", fontSize: 14 }} />
              <Tooltip
                cursor={{ fill: "#f3ede3" }}
                formatter={(v: number) => [`${v}%`, "Score"]}
                contentStyle={{ borderRadius: 8, borderColor: "#e6dfd3", background: "#fffdf9" }}
              />
              <Bar dataKey="pct" radius={[0, 4, 4, 0]} isAnimationActive={false}>
                {data.map((d) => (
                  <Cell key={d.key} fill={d.key === result.predicted_emotion ? "#2f6b65" : "#86bcb5"} />
                ))}
                <LabelList dataKey="pct" position="right" formatter={(v: number) => `${v}%`} style={{ fill: "#55524c", fontSize: 13 }} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <table className="sr-only">
          <caption>Scores by expression</caption>
          <tbody>
            {data.map((d) => (
              <tr key={d.key}>
                <th scope="row">{d.label}</th>
                <td>{d.pct}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </figure>

      <div className="mt-5">
        <p className="mb-2 text-sm font-medium text-ink-soft">Your clip ({result.audio_duration_seconds.toFixed(1)} s)</p>
        <audio controls src={audioUrl} className="w-full" aria-label="Play back your clip" />
      </div>

      <fieldset className="mt-6 rounded-xl border border-line p-4">
        <legend className="px-1 font-medium">Does this match how you feel right now?</legend>
        <div className="mt-2 flex flex-wrap gap-2">
          {(["yes", "partly", "no"] as const).map((v) => (
            <button
              key={v}
              type="button"
              aria-pressed={reflection === v}
              className={reflection === v ? "btn-primary px-3 py-1.5" : "btn-secondary px-3 py-1.5"}
              onClick={() => setReflection(v)}
            >
              {{ yes: "Yes", partly: "Partly", no: "Not really" }[v]}
            </button>
          ))}
        </div>
        {reflection && (
          <p className="mt-3 text-ink-soft" aria-live="polite">
            {REFLECTION_REPLIES[reflection]}
          </p>
        )}
        <p className="mt-2 text-xs text-ink-muted">Your answer here stays on this page and is not sent anywhere.</p>
      </fieldset>

      <p className="mt-5 text-sm text-ink-muted">{result.notice}</p>
      <NotDiagnosisNote />
    </article>
  );
}
