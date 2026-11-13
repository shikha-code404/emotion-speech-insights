import { useEffect, useRef, useState } from "react";
import { getSurveyQuestions, scoreSurvey } from "../services/api";
import { useSession } from "../state/session";
import type { SurveyQuestionsResponse } from "../types/api";
import { ConsentCheckbox, ErrorNotice, NextStepPanel, Spinner } from "../components/common";

const PER_PAGE = 3;

export default function Questionnaire() {
  const { setSurvey, survey, voice } = useSession();
  const [consent, setConsent] = useState(false);
  const [data, setData] = useState<SurveyQuestionsResponse | null>(null);
  const [loadError, setLoadError] = useState<unknown>(null);
  const [answers, setAnswers] = useState<Record<string, number>>({});
  const [page, setPage] = useState(0);
  const [missing, setMissing] = useState<string[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<unknown>(null);
  const [done, setDone] = useState(false);
  const headingRef = useRef<HTMLHeadingElement>(null);

  const load = () => {
    setLoadError(null);
    getSurveyQuestions().then(setData).catch(setLoadError);
  };
  useEffect(load, []);

  const items = data?.items ?? [];
  const pages = Math.max(1, Math.ceil(items.length / PER_PAGE));
  const pageItems = items.slice(page * PER_PAGE, page * PER_PAGE + PER_PAGE);
  const answered = items.filter((i) => answers[i.id] !== undefined).length;
  const isLast = page === pages - 1;

  const goTo = (p: number) => {
    setPage(p);
    setMissing([]);
    requestAnimationFrame(() => headingRef.current?.focus());
  };

  const checkPage = (): boolean => {
    const miss = pageItems.filter((i) => answers[i.id] === undefined).map((i) => i.id);
    setMissing(miss);
    if (miss.length) {
      document.querySelector<HTMLInputElement>(`input[name="${miss[0]}"]`)?.focus();
      return false;
    }
    return true;
  };

  const submit = async () => {
    if (!checkPage()) return;
    const allMissing = items.filter((i) => answers[i.id] === undefined);
    if (allMissing.length) {
      goTo(Math.floor(items.indexOf(allMissing[0]) / PER_PAGE));
      return;
    }
    setSubmitting(true);
    setSubmitError(null);
    try {
      const result = await scoreSurvey({ consent: true, answers });
      setSurvey(result);
      setAnswers({}); // answers are not kept once scored
      setDone(true);
    } catch (e) {
      setSubmitError(e);
    } finally {
      setSubmitting(false);
    }
  };

  if (done && survey) {
    return (
      <div className="space-y-6">
        <h1 className="h1">Wellbeing questionnaire</h1>
        <NextStepPanel
          doneLabel="Questionnaire complete"
          otherDone={!!voice}
          otherTo="/voice"
          otherLabel="Also analyse a voice clip"
        />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="h1">Wellbeing questionnaire</h1>
        <p className="mt-2 text-ink-soft">
          The DASS-21 is a standard set of 21 statements about the past week. It takes about 5 minutes. Your answers give
          a self-report screening result, not a diagnosis.
        </p>
      </div>

      <ConsentCheckbox id="survey-consent" checked={consent} onChange={setConsent}>
        I agree to send my answers to the server to be scored. I understand they are not stored, and that the result is a
        self-report screening result, not a diagnosis.
      </ConsentCheckbox>

      {consent && (
        <section className="card" aria-label="Questions">
          {loadError != null ? (
            <ErrorNotice error={loadError} onRetry={load} />
          ) : !data ? (
            <Spinner label="Loading questions…" />
          ) : (
            <form
              noValidate
              onSubmit={(e) => {
                e.preventDefault();
                if (isLast) void submit();
                else if (checkPage()) goTo(page + 1);
              }}
            >
              <div className="mb-5">
                <div className="mb-1 flex justify-between text-sm text-ink-soft">
                  <span>
                    Page {page + 1} of {pages}
                  </span>
                  <span>
                    {answered} of {items.length} answered
                  </span>
                </div>
                <div
                  className="h-2 overflow-hidden rounded-full bg-sand-200"
                  role="progressbar"
                  aria-label="Questions answered"
                  aria-valuemin={0}
                  aria-valuemax={items.length}
                  aria-valuenow={answered}
                >
                  <div className="h-full bg-teal-500 transition-all" style={{ width: `${(answered / items.length) * 100}%` }} />
                </div>
              </div>

              <h2 ref={headingRef} tabIndex={-1} className="h2 mb-1 outline-none">
                Over the past week…
              </h2>
              <p className="mb-5 text-ink-soft">{data.instructions}</p>

              <div className="space-y-6">
                {pageItems.map((item) => {
                  const isMissing = missing.includes(item.id);
                  return (
                    <fieldset
                      key={item.id}
                      className={`rounded-xl border p-4 ${isMissing ? "border-teal-600 bg-teal-50" : "border-line"}`}
                      aria-describedby={isMissing ? `${item.id}-err` : undefined}
                      data-testid={`item-${item.id}`}
                    >
                      <legend className="px-1 font-medium text-ink">
                        {item.number}. {item.text}
                      </legend>
                      {isMissing && (
                        <p id={`${item.id}-err`} className="mb-2 text-sm font-medium text-teal-800">
                          Please choose an answer for this statement.
                        </p>
                      )}
                      <div className="mt-2 grid gap-2 sm:grid-cols-2">
                        {data.answer_scale.map((opt) => {
                          const id = `${item.id}-${opt.value}`;
                          const checked = answers[item.id] === opt.value;
                          return (
                            <label
                              key={opt.value}
                              htmlFor={id}
                              className={`flex cursor-pointer items-start gap-3 rounded-lg border px-3 py-2.5 ${
                                checked ? "border-teal-600 bg-teal-50" : "border-line hover:bg-sand-100"
                              } has-[:focus-visible]:outline has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-teal-600`}
                            >
                              <input
                                id={id}
                                type="radio"
                                name={item.id}
                                value={opt.value}
                                checked={checked}
                                required
                                onChange={() => {
                                  setAnswers((a) => ({ ...a, [item.id]: opt.value }));
                                  setMissing((m) => m.filter((x) => x !== item.id));
                                }}
                                className="mt-1 h-4 w-4 shrink-0 accent-teal-600"
                              />
                              <span>
                                <span className="font-semibold text-ink">{opt.value}</span>
                                <span className="text-ink-soft"> – {opt.label}</span>
                              </span>
                            </label>
                          );
                        })}
                      </div>
                    </fieldset>
                  );
                })}
              </div>

              {missing.length > 0 && (
                <p role="alert" className="mt-4 font-medium text-teal-800">
                  Every statement needs an answer before you continue.
                </p>
              )}
              {submitError != null && (
                <div className="mt-4">
                  <ErrorNotice error={submitError} />
                </div>
              )}

              <div className="mt-6 flex flex-col-reverse gap-3 sm:flex-row sm:justify-between">
                <button type="button" className="btn-secondary" onClick={() => goTo(page - 1)} disabled={page === 0 || submitting}>
                  Back
                </button>
                <button type="submit" className="btn-primary" disabled={submitting}>
                  {isLast ? (submitting ? "Scoring…" : "See my result") : "Next"}
                </button>
              </div>
            </form>
          )}
        </section>
      )}
    </div>
  );
}
