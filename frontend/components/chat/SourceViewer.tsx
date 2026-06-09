"use client";

import type { Citation, Source } from "@/lib/chat-types";
import { type ReactNode, useEffect, useState } from "react";

type Props = {
  source: Source;
  citation: Citation | null;
  onClose: () => void;
};

/**
 * Right slide-over drawer for a cited source. Shows the retrieved passage
 * (highlighted) and a deep-link to open the filing at the cited page. The real
 * PDF page render (`/sources/{doc_id}`) is wired later; the §9.4 citation
 * already carries page + span, which this presents.
 */
export function SourceViewer({ source, citation, onClose }: Props): ReactNode {
  const [show, setShow] = useState(false);
  // Use `||` not `??`: a citation with an empty span ("") must still fall back to
  // the retrieved snippet, or the drawer body renders blank.
  const passage = citation?.span || source.snippet;
  const page = citation?.page ?? null;

  // Trigger the slide-in after mount; Escape closes.
  useEffect(() => {
    const raf = requestAnimationFrame(() => setShow(true));
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("keydown", onKey);
    };
  }, [onClose]);

  return (
    <>
      <button
        type="button"
        className={`backdrop${show ? " show" : ""}`}
        onClick={onClose}
        aria-label="Close source viewer"
      />
      <aside className={`drawer${show ? " show" : ""}`} aria-label="Source viewer">
        <div className="dhead">
          <div>
            <span className="lbl">Source</span>
            <div className="dt-title">{source.title}</div>
            <div className="dt-meta">
              [{source.id}]{page !== null ? ` · page ${page}` : ""}
            </div>
          </div>
          <button type="button" className="dclose" onClick={onClose} aria-label="Close">
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.7"
              aria-hidden="true"
            >
              <path d="M6 6l12 12M18 6L6 18" />
            </svg>
          </button>
        </div>

        <div className="dbody">
          <div className="dpage">
            {page !== null ? `Page ${page}` : "Retrieved passage"}
            <span className="ln" />
          </div>
          <div className="dquote">
            <mark>{passage}</mark>
          </div>
        </div>

        <a className="dopen" href={source.url} target="_blank" rel="noreferrer">
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.7"
            aria-hidden="true"
          >
            <path d="M14 4h6v6M20 4l-9 9M19 14v5H5V5h5" />
          </svg>
          Open full document at this page
        </a>
      </aside>
    </>
  );
}
