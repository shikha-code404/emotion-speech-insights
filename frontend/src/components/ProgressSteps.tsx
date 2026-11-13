export interface Step {
  label: string;
  detail?: string;
}

/** Ordered progress steps. `current` is the index of the active step; steps before it are done. */
export function ProgressSteps({ steps, current }: { steps: Step[]; current: number }) {
  return (
    <ol className="space-y-3" aria-label="Progress" data-testid="progress-steps">
      {steps.map((s, i) => {
        const status = i < current ? "done" : i === current ? "active" : "todo";
        return (
          <li key={s.label} className="flex items-start gap-3" aria-current={status === "active" ? "step" : undefined}>
            <span
              aria-hidden
              className={`mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-semibold ${
                status === "done"
                  ? "bg-teal-600 text-white"
                  : status === "active"
                    ? "border-2 border-teal-600 text-teal-700"
                    : "border border-line text-ink-muted"
              }`}
            >
              {status === "done" ? "✓" : i + 1}
            </span>
            <span>
              <span className={status === "todo" ? "text-ink-muted" : "font-medium text-ink"}>{s.label}</span>
              <span className="sr-only"> ({status === "done" ? "done" : status === "active" ? "in progress" : "waiting"})</span>
              {s.detail && status === "active" && <span className="block text-sm text-ink-soft">{s.detail}</span>}
            </span>
          </li>
        );
      })}
    </ol>
  );
}
