import type { DocOutcome, UploadView } from "@/lib/monitor-types";
import type { ReactNode } from "react";
import { DocTrace } from "./DocTrace";

const OUTCOME_LABEL: Record<NonNullable<DocOutcome>, string> = {
  indexed: "indexed",
  "facts-written": "facts → DuckDB",
  deferred: "deferred",
  duplicate: "duplicate",
  empty: "no chunks",
  bundled: "bundle member",
  quarantined: "quarantined",
  failed: "failed",
};

function outcomeClass(outcome: DocOutcome): string {
  if (outcome === "indexed" || outcome === "facts-written") return "ok";
  if (outcome === "failed" || outcome === "quarantined") return "bad";
  if (outcome === null) return "run";
  return "muted"; // deferred / duplicate / empty / bundled
}

/** One upload (job): header + a detailed execution trace per file. */
export function UploadCard({ upload }: { upload: UploadView }): ReactNode {
  const complete = upload.docs.every((d) => d.outcome !== null);

  return (
    <div className="ucard">
      <div className="uchead">
        <span className={`ucbadge ${upload.country}`}>{upload.country}</span>
        <span className="ucid mono">{upload.upload_id}</span>
        <span className="ucmeta">
          {upload.docs.length} file{upload.docs.length > 1 ? "s" : ""} · {upload.created}
        </span>
        <span className={`ucstatus ${complete ? "ok" : "run"}`}>
          {complete ? "complete" : "running"}
        </span>
      </div>
      <div className="ucrows">
        {upload.docs.map((doc) => (
          <div key={doc.doc_id} className="ucdoc">
            <div className="ucdochead">
              <span className="ucfile">{doc.filename}</span>
              {doc.outcome && (
                <span className={`ucoutcome ${outcomeClass(doc.outcome)}`}>
                  {OUTCOME_LABEL[doc.outcome]}
                </span>
              )}
            </div>
            <DocTrace doc={doc} />
          </div>
        ))}
      </div>
    </div>
  );
}
