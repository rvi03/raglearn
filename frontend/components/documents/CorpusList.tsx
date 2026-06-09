"use client";

import { useUploads } from "@/hooks/useUploads";
import type { Country } from "@/lib/ingest-types";
import { type ReactNode, useMemo, useState } from "react";

/** Map a document's status to the shared outcome-badge style. */
function badgeClass(status: string): string {
  if (status === "indexed" || status === "facts-written") return "ok";
  if (status === "quarantined" || status === "failed") return "bad";
  if (status === "pending" || status === "processing") return "run";
  return "muted"; // duplicate / empty / bundled / deferred
}

function baseName(path: string): string {
  const i = path.lastIndexOf("/");
  return i === -1 ? path : path.slice(i + 1);
}

type Row = {
  country: Country;
  file: string; // basename
  path: string; // full object key (doc_id)
  uploaded: string;
  status: string;
  detail: string | null;
  key: string;
};

/**
 * Persistent corpus — a searchable, filterable table of every uploaded document
 * and its ingestion status. Reads the durable store (polled), so status ticks
 * pending → processing → indexed without needing to watch the run live.
 */
export function CorpusList(): ReactNode {
  const uploads = useUploads();
  const [query, setQuery] = useState("");
  const [country, setCountry] = useState<"all" | Country>("all");
  const [status, setStatus] = useState("all");

  const rows = useMemo<Row[]>(
    () =>
      uploads.flatMap((u) =>
        u.docs.map((d) => ({
          country: u.country,
          file: baseName(d.filename),
          path: d.doc_id,
          uploaded: u.created,
          status: d.status,
          detail: d.detail,
          key: d.doc_id,
        })),
      ),
    [uploads],
  );

  const statuses = useMemo(() => [...new Set(rows.map((r) => r.status))].sort(), [rows]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return rows.filter(
      (r) =>
        (country === "all" || r.country === country) &&
        (status === "all" || r.status === status) &&
        (q === "" || r.file.toLowerCase().includes(q) || r.path.toLowerCase().includes(q)),
    );
  }, [rows, query, country, status]);

  return (
    <div className="corpus">
      <div className="corpus-controls">
        <input
          className="corpus-search"
          type="search"
          placeholder="Search file or path…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Search documents"
        />
        <select
          value={country}
          onChange={(e) => setCountry(e.target.value as "all" | Country)}
          aria-label="Filter by country"
        >
          <option value="all">All countries</option>
          <option value="us">US</option>
          <option value="india">India</option>
        </select>
        <select
          value={status}
          onChange={(e) => setStatus(e.target.value)}
          aria-label="Filter by status"
        >
          <option value="all">All statuses</option>
          {statuses.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
        <span className="corpus-count">
          {filtered.length} of {rows.length}
        </span>
      </div>

      {rows.length === 0 ? (
        <div className="empty">
          <div className="ekick">finrag · corpus</div>
          <h2>No documents yet</h2>
          <p>Uploaded filings and their ingestion status appear here.</p>
        </div>
      ) : (
        <table className="corpus-table">
          <thead>
            <tr>
              <th>Country</th>
              <th>File</th>
              <th>Path</th>
              <th>Uploaded</th>
              <th>Status</th>
              <th>Detail</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r) => (
              <tr key={r.key}>
                <td>
                  <span className={`ucbadge ${r.country}`}>{r.country}</span>
                </td>
                <td className="ct-file" title={r.file}>
                  {r.file}
                </td>
                <td className="ct-path mono" title={r.path}>
                  {r.path}
                </td>
                <td className="ct-when">{r.uploaded?.slice(0, 16).replace("T", " ")}</td>
                <td>
                  <span className={`ucoutcome ${badgeClass(r.status)}`}>{r.status}</span>
                </td>
                <td className="ct-detail">{r.detail ?? ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
