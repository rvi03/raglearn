"use client";

import { useMonitor } from "@/hooks/useMonitor";
import { type ReactNode, useMemo, useState } from "react";
import { UploadRow } from "./UploadRow";

/** Monitor — the live ingestion pipeline DAG (per-upload, per-document trace). */
export function MonitorView(): ReactNode {
  const uploads = useMonitor();
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState<"all" | "running" | "complete">("all");

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return uploads.filter((u) => {
      const complete = u.docs.every((d) => d.outcome !== null);
      if (status === "running" && complete) return false;
      if (status === "complete" && !complete) return false;
      if (q === "") return true;
      return (
        u.upload_id.toLowerCase().includes(q) ||
        u.docs.some((d) => d.filename.toLowerCase().includes(q))
      );
    });
  }, [uploads, query, status]);

  return (
    <>
      <div className="mhead">
        <div className="ctx">
          <div className="fic">MON</div>
          <div className="nm">
            Monitor
            <small>ingestion activity</small>
          </div>
        </div>
        <div className="mtools">
          <div className="mode">
            <span className="d" />
            Live
          </div>
        </div>
      </div>

      <div className="scroll">
        {uploads.length === 0 ? (
          <div className="empty">
            <div className="ekick">finrag · monitor</div>
            <h2>No uploads yet</h2>
            <p>Ingestion runs appear here as a pipeline DAG. Click one for its full trace.</p>
          </div>
        ) : (
          <div className="urwrap">
            <div className="corpus-controls">
              <input
                className="corpus-search"
                type="search"
                placeholder="Search upload id or file…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                aria-label="Search uploads"
              />
              <select
                value={status}
                onChange={(e) => setStatus(e.target.value as "all" | "running" | "complete")}
                aria-label="Filter by status"
              >
                <option value="all">All statuses</option>
                <option value="running">Running</option>
                <option value="complete">Complete</option>
              </select>
              <span className="corpus-count">
                {filtered.length} of {uploads.length}
              </span>
            </div>

            <div className="urlist">
              <div className="urhead">
                <span />
                <span className="lbl">Upload</span>
                <span className="lbl">Files</span>
                <span className="lbl">Flow</span>
                <span className="lbl">Status</span>
                <span className="lbl">When</span>
                <span />
              </div>
              {filtered.map((u) => (
                <UploadRow key={u.upload_id} upload={u} />
              ))}
            </div>
          </div>
        )}
      </div>
    </>
  );
}
