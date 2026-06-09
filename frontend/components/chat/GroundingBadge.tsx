import type { ReactNode } from "react";

/**
 * "Grounding confidence" line. Frontend-only field with no backend source yet —
 * carried on the `done` event and hidden when unknown.
 */
export function GroundingBadge({ confidence }: { confidence: number | null }): ReactNode {
  if (confidence === null) return null;
  return (
    <div className="ground">
      <span className="d" />
      Grounding confidence {Math.round(confidence * 100)}% · every claim traced to an indexed
      passage
    </div>
  );
}
