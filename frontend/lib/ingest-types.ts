/**
 * Frontend types for the document upload seam (backend `POST /ingest/upload`).
 * Mirrors the FastAPI `UploadResponse` shape.
 */

export type Country = "us" | "india";

/** A file staged in the UI before submit; `path` is its relative path. */
export type StagedFile = {
  id: string;
  file: File;
  path: string;
  size: number;
};

/** One object written by the backend. */
export type UploadedDoc = {
  doc_id: string;
  filename: string;
  size: number;
  content_type: string;
};

export type UploadResponse = {
  upload_id: string;
  country: Country;
  docs: UploadedDoc[];
};

export type UploadStatus = "idle" | "uploading" | "done" | "error";

/** A file plus its relative path, as gathered from a drop or folder picker. */
export type PathFile = { file: File; path: string };
