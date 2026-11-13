import { useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import { friendlyError } from "../lib/errors";

export function NotDiagnosisNote() {
  return (
    <p className="mt-5 rounded-lg border border-line bg-sand-100 px-4 py-3 text-sm text-ink-soft" data-testid="not-diagnosis-note">
      <span className="font-semibold text-ink">This is not a diagnosis.</span> If you are concerned about how you
      are feeling, please speak to a qualified professional.
    </p>
  );
}

export function ConsentCheckbox({
  id,
  checked,
  onChange,
  children,
}: {
  id: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  children: React.ReactNode;
}) {
  return (
    <div className="card">
      <div className="flex items-start gap-3">
        <input
          id={id}
          type="checkbox"
          className="mt-1 h-5 w-5 shrink-0 accent-teal-600"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
        />
        <label htmlFor={id} className="text-ink-soft">
          {children}
        </label>
      </div>
    </div>
  );
}

export function ErrorNotice({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const { code, title, hint } = friendlyError(error);
  return (
    <div role="alert" className="rounded-xl border border-sand-200 bg-sand-100 px-4 py-3" data-error-code={code}>
      <p className="font-semibold text-ink">{title}</p>
      <p className="text-ink-soft">{hint}</p>
      {onRetry && (
        <button type="button" className="btn-ghost mt-2 -ml-2 px-2 py-1" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}

export function NextStepPanel({
  doneLabel,
  otherDone,
  otherTo,
  otherLabel,
}: {
  doneLabel: string;
  otherDone: boolean;
  otherTo: string;
  otherLabel: string;
}) {
  // The control the user pressed has just disappeared; move keyboard focus to the next step.
  const viewRef = useRef<HTMLAnchorElement>(null);
  useEffect(() => viewRef.current?.focus(), []);
  return (
    <div className="card border-teal-200 bg-teal-50" role="status">
      <p className="h2 text-teal-800">{doneLabel}</p>
      <p className="mt-1 text-ink-soft">Your result is ready. It is kept only in this browser tab until you leave the results page.</p>
      <div className="mt-4 flex flex-col gap-3 sm:flex-row">
        <Link ref={viewRef} to="/results" className="btn-primary">
          View results
        </Link>
        {!otherDone && (
          <Link to={otherTo} className="btn-secondary">
            {otherLabel} (optional)
          </Link>
        )}
      </div>
    </div>
  );
}

export function Spinner({ label }: { label: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-ink-soft" role="status">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-teal-300 border-t-teal-700" aria-hidden />
      {label}
    </span>
  );
}
