"use client";

import type { Country } from "@/lib/ingest-types";
import type { DocOutcome, NodeStatus, UploadView } from "@/lib/monitor-types";
import { useEffect, useState } from "react";

/**
 * Drive the monitor DAG from the **durable** ingestion store (the per-document
 * execution trace), polled. This makes the DAG (MiniDag + DocTrace) render at any
 * time — including after a run — and tick as the consumer works, rather than the
 * old live-only Redis SSE that left the page blank once a run finished.
 */

type ApiTraceNode = { id: string; label: string; status: string; detail: string | null };
type ApiDoc = { doc_id: string; filename: string; outcome: string | null; trace: ApiTraceNode[] };
type ApiUpload = { upload_id: string; country: Country; created: string; docs: ApiDoc[] };

function toView(u: ApiUpload): UploadView {
  return {
    upload_id: u.upload_id,
    country: u.country,
    created: u.created,
    docs: u.docs.map((d) => ({
      doc_id: d.doc_id,
      filename: d.filename,
      outcome: (d.outcome as DocOutcome) ?? null,
      trace: d.trace.map((t) => ({
        id: t.id,
        label: t.label,
        detail: t.detail ?? undefined,
        status: t.status as NodeStatus,
      })),
    })),
  };
}

export function useMonitor(pollMs = 3000): UploadView[] {
  const [uploads, setUploads] = useState<UploadView[]>([]);

  useEffect(() => {
    let alive = true;
    const tick = async (): Promise<void> => {
      try {
        const res = await fetch("/api/ingestion/uploads", { cache: "no-store" });
        if (!res.ok) return;
        const data = (await res.json()) as { uploads?: ApiUpload[] };
        if (alive) setUploads((data.uploads ?? []).map(toView));
      } catch {
        // transient; keep the last good view and retry next tick
      }
    };
    void tick();
    const id = setInterval(tick, pollMs);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [pollMs]);

  return uploads;
}
