import { Link } from "react-router-dom";

export default function Home() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="h1">Voice &amp; Wellbeing Check</h1>
        <p className="mt-3 max-w-2xl text-lg text-ink-soft">
          Hear what emotion a short voice clip expresses, or reflect on your past week with a standard self-report
          questionnaire. Each one works on its own.
        </p>
      </section>

      <section aria-label="Privacy" className="rounded-xl border border-teal-200 bg-teal-50 px-4 py-3 text-teal-800">
        <p>
          <span className="font-semibold">Your privacy:</span> recordings are deleted on the server right after
          analysis, questionnaire answers are never stored, and your results disappear when you leave the results page.{" "}
          <Link to="/privacy" className="underline underline-offset-2">
            Read the privacy details
          </Link>
          .
        </p>
      </section>

      <section aria-label="Choose a check" className="grid gap-5 md:grid-cols-2">
        <div className="card flex flex-col">
          <h2 className="h2">Voice check</h2>
          <p className="mt-2 flex-1 text-ink-soft">
            Record 5–15 seconds of speech or upload a clip. A model estimates which emotion your voice expresses in
            that clip.
          </p>
          <Link to="/voice" className="btn-primary mt-5 self-start">
            Analyse a voice clip
          </Link>
        </div>
        <div className="card flex flex-col">
          <h2 className="h2">Wellbeing questionnaire</h2>
          <p className="mt-2 flex-1 text-ink-soft">
            Answer 21 short statements about the past week (the DASS-21). You get a self-report screening result from
            your own answers.
          </p>
          <Link to="/questionnaire" className="btn-primary mt-5 self-start">
            Take the wellbeing questionnaire
          </Link>
        </div>
      </section>

      <p className="text-ink-muted">
        Doing both is optional. The two results are always shown separately and never combined. Neither one is a
        diagnosis.
      </p>
    </div>
  );
}
