/**
 * Types for the persistent corpus / uploads list (backend `GET /ingestion/uploads`).
 * Unlike the live monitor SSE (no history), this is the durable record of what was
 * uploaded and each document's ingestion status — read on demand and polled.
 */

import type { Country } from "@/lib/ingest-types";

/** One uploaded document and its durable ingestion status. */
export type CorpusDoc = {
  doc_id: string;
  filename: string;
  stage: string | null; // last pipeline stage seen (detect/parse/…)
  stage_status: string | null; // that stage's status (running/done/…)
  outcome: string | null; // terminal outcome once finished
  status: string; // display status: outcome ?? processing ?? pending
  detail: string | null;
};

/** An upload batch with its documents. */
export type CorpusUpload = {
  upload_id: string;
  country: Country;
  created: string;
  docs: CorpusDoc[];
};
