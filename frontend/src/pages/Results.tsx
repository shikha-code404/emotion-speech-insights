import { useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useSession } from "../state/session";
import { VoiceResultCard } from "../components/VoiceResultCard";
import { SurveyResultCard } from "../components/SurveyResultCard";

export default function Results() {
  const { voice, survey, modelInfo, loadModelInfo } = useSession();
  const navigate = useNavigate();

  useEffect(() => {
    if (voice) loadModelInfo().catch(() => undefined); // label order; card falls back to response order
  }, [voice, loadModelInfo]);

  if (!voice && !survey) {
    return (
      <div className="space-y-5">
        <h1 className="h1">Results</h1>
        <div className="card">
          <p className="text-ink-soft">
            There are no results to show. Results are kept only until you leave this page, and nothing is saved.
          </p>
          <div className="mt-4 flex flex-col gap-3 sm:flex-row">
            <Link to="/voice" className="btn-primary">
              Analyse a voice clip
            </Link>
            <Link to="/questionnaire" className="btn-secondary">
              Take the wellbeing questionnaire
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="h1">Your results</h1>
        <p className="mt-2 text-ink-soft">
          {voice && survey
            ? "These are two separate checks that measure different things, so they are shown separately and never combined."
            : "Here is the result of the check you completed."}{" "}
          They will be cleared when you leave this page.
        </p>
      </div>

      <div className={voice && survey ? "grid gap-6 lg:grid-cols-2 lg:items-start" : "grid gap-6"}>
        {voice && <VoiceResultCard result={voice.result} audioUrl={voice.audioUrl} labels={modelInfo?.labels ?? null} />}
        {survey && <SurveyResultCard result={survey} />}
      </div>

      <div className="flex flex-col gap-3 sm:flex-row">
        {!voice && (
          <Link to="/voice" className="btn-secondary">
            Also analyse a voice clip (clears this result)
          </Link>
        )}
        {!survey && (
          <Link to="/questionnaire" className="btn-secondary">
            Also take the questionnaire (clears this result)
          </Link>
        )}
        <button type="button" className="btn-primary" onClick={() => navigate("/")}>
          Finish and clear results
        </button>
      </div>
    </div>
  );
}
