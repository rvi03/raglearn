/**
 * Frontend mirror of the backend SSE contract and chat domain types.
 * This is the frozen seam the frontend codes against — when the real `/chat`
 * backend lands, only the transport swaps; these shapes stay.
 *
 * Note: `done.grounding_confidence` is a frontend-only field with no backend
 * source yet; it is carried on `done` and rendered when present.
 */

export type StepStatus = "running" | "done" | "failed";

/** A single Server-Sent Event, discriminated by `type`. */
export type SSEEvent =
  | { type: "token"; delta: string }
  | { type: "citation"; id: number; source_doc_id: string; page: number; span: string }
  | { type: "source"; id: number; title: string; url: string; snippet: string }
  | {
      type: "agent_step";
      name: string;
      path: string;
      detail?: string;
      status: StepStatus;
      latency_ms: number;
      tokens_in: number;
      tokens_out: number;
      model: string;
      usd: number;
    }
  | {
      type: "done";
      usage: { tokens_in: number; tokens_out: number };
      trace_id: string;
      query_usd: number;
      grounding_confidence?: number;
      /** False when a guard refused the answer (safety). */
      answered?: boolean;
      /** PII entity types masked in the answer, e.g. ["EMAIL", "IN_PAN"]. */
      redacted?: string[];
    };

/** Resolved citation marker `[n]` → where it points in the source doc. */
export type Citation = {
  id: number;
  source_doc_id: string;
  page: number;
  span: string;
};

/** Evidence card shown in-thread and in the source viewer. */
export type Source = {
  id: number;
  title: string;
  url: string;
  snippet: string;
};

export type AgentStep = {
  name: string;
  path: string;
  detail: string;
  status: StepStatus;
  latency_ms: number;
  tokens_in: number;
  tokens_out: number;
  model: string;
  usd: number;
};

export type MessageRole = "user" | "assistant";

/** A persisted conversation as listed in the Recent rail. */
export type Session = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
};

/** One persisted turn as returned when a conversation is reopened. */
export type StoredMessage = {
  id: string;
  role: MessageRole;
  text: string;
  /** Assistant turns carry the rendered evidence/grounding/cost; user turns are null. */
  meta: Record<string, unknown> | null;
  created_at: string;
};

/** One turn in the thread. The assistant turn fills as SSE events arrive. */
export type ChatMessage = {
  id: string;
  role: MessageRole;
  text: string;
  citations: Citation[];
  sources: Source[];
  steps: AgentStep[];
  /** Set on the `done` event. */
  costUsd: number | null;
  traceId: string | null;
  groundingConfidence: number | null;
  /** False when a guard refused the answer; defaults true until `done`. */
  answered: boolean;
  /** PII entity types masked in the answer (empty when none). */
  redacted: string[];
  streaming: boolean;
};
