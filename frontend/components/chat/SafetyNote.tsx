import type { ReactNode } from "react";

/**
 * Surfaces the answer's safety/privacy signals from the `done` event: a refusal
 * when a guard withheld the answer, and a note listing any PII types masked.
 * Hidden when the answer was clean and fully shown.
 */
export function SafetyNote({
  answered,
  redacted,
}: {
  answered: boolean;
  redacted: string[];
}): ReactNode {
  if (answered && redacted.length === 0) return null;

  return (
    <div className="safety">
      {!answered && (
        <span className="safety-flag refused">
          <span className="d" />
          Answer withheld by a safety guard
        </span>
      )}
      {redacted.length > 0 && (
        <span className="safety-flag redacted">
          <span className="d" />
          PII redacted: {redacted.join(", ")}
        </span>
      )}
    </div>
  );
}
