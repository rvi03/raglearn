/**
 * Types for the upload-monitor view. The DAG is a *dynamic execution trace* per
 * document — the pipeline branches (facts vs pages, deferred bundles, skips,
 * quarantine), so which nodes appear depends on the path taken. The backend
 * (Redis stage emitter → SSE) emits these events.
 */

import type { Country } from "@/lib/ingest-types";

export type NodeStatus = "running" | "done" | "skipped" | "deferred" | "failed";

/** One step in a document's path, with the concrete branch decision. */
export type TraceNode = {
  id: string; // unique within a doc, e.g. "detect" | "route" | "parse" | "extract"
  label: string; // the action, e.g. "Detect", "Parse"
  detail?: string; // the decision/result, e.g. "PDF", "Docling vision"
  status: NodeStatus;
};

/**
 * Terminal result of a document's run — the backend's `doc_done` outcome.
 * The three skip reasons are kept distinct (not collapsed to one "skipped"): a
 * `duplicate` re-upload, an `empty` parse, and an XBRL `bundled` member are very
 * different operator signals.
 */
export type DocOutcome =
  | "indexed" // pages arm wrote chunks
  | "facts-written" // facts arm wrote facts to DuckDB
  | "deferred" // facts arm: XBRL bundle still incomplete
  | "duplicate" // pages arm: content-hash dedup hit
  | "empty" // pages arm: parsed but produced no chunks
  | "bundled" // XBRL bundle member, pulled in with its instance
  | "quarantined" // unrecognized format
  | "failed" // errored mid-run
  | null; // still running

export type DocView = {
  doc_id: string;
  filename: string;
  trace: TraceNode[]; // ordered, grows as events arrive
  outcome: DocOutcome;
};

export type UploadView = {
  upload_id: string;
  country: Country;
  created: string;
  docs: DocView[];
};

/** Major stages shown in the high-level mini-DAG (a glanceable row summary). */
export const MACRO_STAGES: readonly { id: string; label: string }[] = [
  { id: "detect", label: "detect" },
  { id: "parse", label: "parse" },
  { id: "chunk", label: "chunk" },
  { id: "embed", label: "embed" },
  { id: "index", label: "index" },
];

/** Fold the detailed trace nodes into the macro stages above. */
export const NODE_TO_MACRO: Record<string, string> = {
  detect: "detect",
  route: "detect",
  parse: "parse",
  bundle: "parse",
  extract: "parse",
  identify: "parse",
  chunk: "chunk",
  embed: "embed",
  index: "index",
  write: "index",
};

/** Aggregate status of a macro stage across an upload's documents. */
export type MacroStatus = NodeStatus | "pending" | "na";

export type MacroNode = { id: string; label: string; status: MacroStatus };

/** Wire events from the monitor stream (the frozen seam). */
export type MonitorEvent =
  | {
      type: "upload";
      upload_id: string;
      country: Country;
      created: string;
      docs: { doc_id: string; filename: string }[];
    }
  | {
      type: "node";
      upload_id: string;
      doc_id: string;
      id: string;
      label: string;
      detail?: string;
      status: NodeStatus;
    }
  | { type: "doc_done"; upload_id: string; doc_id: string; outcome: DocOutcome };
