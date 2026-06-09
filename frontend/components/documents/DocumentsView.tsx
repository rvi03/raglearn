"use client";

import { type ReactNode, useState } from "react";
import { CorpusList } from "./CorpusList";
import { UploadZone } from "./UploadZone";

type Tab = "upload" | "files";

/** Documents view — two tabs: Upload (entry) and Files (persistent corpus). */
export function DocumentsView(): ReactNode {
  const [tab, setTab] = useState<Tab>("upload");

  return (
    <>
      <div className="mhead">
        <div className="ctx">
          <div className="fic">DOC</div>
          <div className="nm">
            Documents
            <small>upload filings → ingestion</small>
          </div>
        </div>
        <div className="doctabs" role="tablist">
          <button
            type="button"
            role="tab"
            aria-selected={tab === "upload"}
            className={tab === "upload" ? "doctab on" : "doctab"}
            onClick={() => setTab("upload")}
          >
            Upload
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={tab === "files"}
            className={tab === "files" ? "doctab on" : "doctab"}
            onClick={() => setTab("files")}
          >
            Files
          </button>
        </div>
      </div>

      <div className="scroll">{tab === "upload" ? <UploadZone /> : <CorpusList />}</div>
    </>
  );
}
