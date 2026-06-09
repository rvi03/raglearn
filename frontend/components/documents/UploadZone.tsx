"use client";

import { useUpload } from "@/hooks/useUpload";
import { gatherDropped } from "@/lib/dropzone";
import type { Country } from "@/lib/ingest-types";
import { type DragEvent, type ReactNode, useEffect, useRef, useState } from "react";

function fmtSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const HINT: Record<Country, string> = {
  us: "US filing — include the inline-XBRL instance and its .xsd (drop the whole filing folder).",
  india: "India filing — a single PDF (or files) is consumed as-is.",
};

export function UploadZone(): ReactNode {
  const {
    country,
    files,
    status,
    result,
    error,
    setCountry,
    addPathFiles,
    addFromInput,
    remove,
    clear,
    submit,
  } = useUpload();
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);

  // `webkitdirectory`/`directory` aren't typed input attributes — set on the ref.
  useEffect(() => {
    const el = folderInput.current;
    if (el) {
      el.setAttribute("webkitdirectory", "");
      el.setAttribute("directory", "");
    }
  }, []);

  const onDrop = async (e: DragEvent<HTMLDivElement>): Promise<void> => {
    e.preventDefault();
    setDragging(false);
    const gathered = await gatherDropped(e.dataTransfer.items);
    addPathFiles(gathered);
  };

  return (
    <div className="uploadwrap">
      <div className="upfield">
        <span className="lbl">Country</span>
        <select
          value={country}
          onChange={(e) => setCountry(e.target.value as Country)}
          aria-label="Country"
        >
          <option value="" disabled>
            Select country…
          </option>
          <option value="india">India</option>
          <option value="us">United States</option>
        </select>
      </div>

      <div
        className={`dropzone${dragging ? " drag" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <div className="dzicon">↑</div>
        <div className="dztitle">Drop a filing folder or files</div>
        <div className="dzhint">{country ? HINT[country] : "Select a country to begin."}</div>
        <div className="upbtns">
          <button type="button" className="upbtn" onClick={() => fileInput.current?.click()}>
            Add files
          </button>
          <button type="button" className="upbtn" onClick={() => folderInput.current?.click()}>
            Add folder
          </button>
        </div>
        <input
          ref={fileInput}
          type="file"
          multiple
          hidden
          onChange={(e) => {
            addFromInput(e.target.files, false);
            e.target.value = "";
          }}
        />
        <input
          ref={folderInput}
          type="file"
          hidden
          onChange={(e) => {
            addFromInput(e.target.files, true);
            e.target.value = "";
          }}
        />
      </div>

      {files.length > 0 && (
        <div className="staged">
          <div className="stagedhead">
            <span className="lbl">
              {files.length} file{files.length > 1 ? "s" : ""} staged
            </span>
            <button type="button" className="upclear" onClick={clear}>
              Clear
            </button>
          </div>
          {files.map((f) => (
            <div key={f.id} className="stagedrow">
              <span className="stagedpath">{f.path}</span>
              <span className="stagedsize">{fmtSize(f.size)}</span>
              <button
                type="button"
                className="stagedx"
                onClick={() => remove(f.id)}
                aria-label={`Remove ${f.path}`}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="uploadbar">
        <button
          type="button"
          className="uploadbtn"
          onClick={submit}
          disabled={files.length === 0 || !country || status === "uploading"}
        >
          {status === "uploading" ? "Uploading…" : "Upload"}
        </button>
        {status === "done" && result && (
          <span className="upok">
            ✓ {result.docs.length} file{result.docs.length > 1 ? "s" : ""} uploaded ·{" "}
            <span className="mono">
              {result.country}/{result.upload_id.slice(0, 8)}…
            </span>
          </span>
        )}
        {status === "error" && error && <span className="uperr">{error}</span>}
      </div>
    </div>
  );
}
