"use client";

import type { CorpusUpload } from "@/lib/corpus-types";
import { useEffect, useState } from "react";

/**
 * Poll the persistent corpus list. Unlike the live monitor SSE, this works at any
 * time (durable status), and polling makes status tick pending → processing →
 * indexed as the consumer works — without needing the tab open during the run.
 */
export function useUploads(pollMs = 4000): CorpusUpload[] {
  const [uploads, setUploads] = useState<CorpusUpload[]>([]);

  useEffect(() => {
    let alive = true;
    const tick = async (): Promise<void> => {
      try {
        const res = await fetch("/api/ingestion/uploads", { cache: "no-store" });
        if (!res.ok) return;
        const data = (await res.json()) as { uploads?: CorpusUpload[] };
        if (alive) setUploads(data.uploads ?? []);
      } catch {
        // transient; keep the last good list and retry next tick
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
