"use client";

import { useMonitor } from "@/hooks/useMonitor";
import Link from "next/link";
import type { ReactNode } from "react";
import { UploadCard } from "./UploadCard";

/** Detail page for one upload — the full per-document execution traces. */
export function UploadDetail({ uploadId }: { uploadId: string }): ReactNode {
  const uploads = useMonitor();
  const upload = uploads.find((u) => u.upload_id === uploadId);

  return (
    <>
      <div className="mhead">
        <div className="ctx">
          <Link href="/monitor" className="backlink" aria-label="Back to monitor">
            ←
          </Link>
          <div className="fic">RUN</div>
          <div className="nm">
            <span className="mono">{uploadId}</span>
            <small>upload trace</small>
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
        {upload ? (
          <div className="monlist">
            <UploadCard upload={upload} />
          </div>
        ) : (
          <div className="empty">
            <div className="ekick">finrag · monitor</div>
            <h2>Locating run…</h2>
            <p>Waiting for events for this upload.</p>
          </div>
        )}
      </div>
    </>
  );
}
