"use client";

import type {
  Country,
  PathFile,
  StagedFile,
  UploadResponse,
  UploadStatus,
} from "@/lib/ingest-types";
import { useCallback, useState } from "react";

// OS / VCS junk that should never be uploaded.
const JUNK = /(^|\/)(\.DS_Store|__MACOSX|Thumbs\.db|\.git)(\/|$)/;

function stage(file: File, path: string): StagedFile {
  return { id: crypto.randomUUID(), file, path, size: file.size };
}

// "" = no country chosen yet — the user must pick one before uploading (no default).
export type CountryChoice = Country | "";

export type UseUpload = {
  country: CountryChoice;
  files: StagedFile[];
  status: UploadStatus;
  result: UploadResponse | null;
  error: string | null;
  setCountry: (c: CountryChoice) => void;
  addPathFiles: (items: PathFile[]) => void;
  addFromInput: (list: FileList | null, fromFolder: boolean) => void;
  remove: (id: string) => void;
  clear: () => void;
  submit: () => Promise<void>;
};

export function useUpload(): UseUpload {
  const [country, setCountry] = useState<CountryChoice>("");
  const [files, setFiles] = useState<StagedFile[]>([]);
  const [status, setStatus] = useState<UploadStatus>("idle");
  const [result, setResult] = useState<UploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const addPathFiles = useCallback((items: PathFile[]): void => {
    const fresh = items.filter((i) => !JUNK.test(i.path)).map((i) => stage(i.file, i.path));
    setFiles((prev) => {
      const seen = new Set(prev.map((p) => p.path));
      return [...prev, ...fresh.filter((f) => !seen.has(f.path))];
    });
    setStatus("idle");
    setResult(null);
    setError(null);
  }, []);

  const addFromInput = useCallback(
    (list: FileList | null, fromFolder: boolean): void => {
      if (!list) return;
      const items = Array.from(list).map((file) => ({
        file,
        path: fromFolder ? file.webkitRelativePath || file.name : file.name,
      }));
      addPathFiles(items);
    },
    [addPathFiles],
  );

  const remove = useCallback((id: string): void => {
    setFiles((prev) => prev.filter((f) => f.id !== id));
  }, []);

  const clear = useCallback((): void => {
    setFiles([]);
    setResult(null);
    setError(null);
    setStatus("idle");
  }, []);

  const submit = useCallback(async (): Promise<void> => {
    if (files.length === 0 || !country || status === "uploading") return;
    setStatus("uploading");
    setError(null);

    const form = new FormData();
    form.append("country", country);
    // Third arg sets the multipart filename to the relative path — the backend
    // reads it as the object key suffix, preserving folder structure.
    for (const f of files) form.append("files", f.file, f.path);

    try {
      const res = await fetch("/ingest/upload", { method: "POST", body: form });
      if (!res.ok) {
        const body = (await res.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(body?.detail ?? `upload failed (${res.status})`);
      }
      const data = (await res.json()) as UploadResponse;
      setResult(data);
      setStatus("done");
      setFiles([]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "upload failed");
      setStatus("error");
    }
  }, [files, country, status]);

  return {
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
  };
}
