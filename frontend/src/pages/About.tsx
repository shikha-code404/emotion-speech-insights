import { useEffect } from "react";
import { useSession } from "../state/session";
import { ErrorNotice, Spinner } from "../components/common";
import { capitalise } from "../lib/errors";

function pct(v: unknown): string {
  return typeof v === "number" ? `${Math.round(v * 100)}%` : "–";
}

export default function About() {
  const { modelInfo, modelInfoError, loadModelInfo } = useSession();
  useEffect(() => {
    loadModelInfo().catch(() => undefined);
  }, [loadModelInfo]);

  const m = modelInfo?.metrics ?? {};

  return (
    <div className="space-y-8">
      <div>
        <h1 className="h1">How it works</h1>
        <p className="mt-2 text-ink-soft">
          This site has two separate modules. They measure different things, use different methods, and their results are
          never combined.
        </p>
      </div>

      {modelInfoError != null && !modelInfo && (
        <ErrorNotice error={modelInfoError} onRetry={() => loadModelInfo().catch(() => undefined)} />
      )}
      {!modelInfo && modelInfoError == null && <Spinner label="Loading model information…" />}

      <section className="card prose-body" aria-labelledby="about-voice">
        <h2 id="about-voice" className="h2 mb-3">
          1. Voice check
        </h2>
        <p>
          Your clip is converted to 16 kHz mono audio, silence at the start and end is trimmed, and the volume is
          normalised. The model then measures acoustic features such as pitch-related spectral shape, energy and timbre, and
          estimates which emotion the voice expresses in that clip.
        </p>
        {modelInfo && (
          <>
            <p>
              <span className="font-semibold text-ink">Model:</span> {modelInfo.model_family} ({modelInfo.version}).
            </p>
            <p>
              <span className="font-semibold text-ink">Expressions it can recognise:</span>{" "}
              {modelInfo.labels.map(capitalise).join(", ")}.
            </p>
            <p>
              <span className="font-semibold text-ink">Training data:</span> acted speech only, from{" "}
              {modelInfo.corpora.join(" and ")}. Actors were asked to perform each emotion, so real everyday speech is usually
              subtler.
            </p>
            <p>
              <span className="font-semibold text-ink">How well it does on actors it has never heard:</span> balanced
              accuracy {pct(m["unweighted_average_recall"])}, macro-F1 {pct(m["macro_f1"])} (
              {String(m["evaluation"] ?? "held-out speakers")}). This is not a measure of real-world accuracy.
            </p>
          </>
        )}
      </section>

      <section className="card prose-body" aria-labelledby="about-survey">
        <h2 id="about-survey" className="h2 mb-3">
          2. Wellbeing questionnaire
        </h2>
        <p>
          The questionnaire is the {modelInfo?.survey.instrument ?? "DASS-21"}. You rate 21 statements about the past week
          from 0 to 3. Seven statements each make up the Depression, Anxiety and Stress scales; each scale's total is doubled
          (0–42) and placed in a standard severity band from Normal to Extremely Severe.
        </p>
        <p>
          The result comes only from your own answers, using a fixed scoring formula with no machine learning. It is a
          self-report screening result, not a diagnosis.
        </p>
        <p>
          <span className="font-semibold text-ink">Development data:</span> the dataset used to build and test this module
          {modelInfo ? (modelInfo.survey.development_data_synthetic ? " is fully synthetic" : " is real survey data") : " is fully synthetic"}{" "}
          (artificially generated respondents, not real people).
        </p>
      </section>

      {modelInfo && (
        <section className="card" aria-labelledby="about-limits">
          <h2 id="about-limits" className="h2 mb-3">
            Known limitations
          </h2>
          <h3 className="mb-2 font-semibold">Voice check</h3>
          <ul className="mb-5 list-disc space-y-1 pl-5 text-ink-soft" data-testid="voice-limitations">
            {modelInfo.limitations.map((l) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
          <h3 className="mb-2 font-semibold">Wellbeing questionnaire</h3>
          <ul className="list-disc space-y-1 pl-5 text-ink-soft" data-testid="survey-limitations">
            {modelInfo.survey.limitations.map((l) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
