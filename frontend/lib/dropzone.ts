import type { PathFile } from "@/lib/ingest-types";

/**
 * Gather files from a drag-drop, descending into dropped folders and preserving
 * each file's relative path (so a filing bundle's structure survives upload).
 * Uses the File System Entries API (`webkitGetAsEntry`), the only way to read a
 * dropped *folder* in the browser.
 */
export async function gatherDropped(items: DataTransferItemList): Promise<PathFile[]> {
  const roots: FileSystemEntry[] = [];
  for (const item of Array.from(items)) {
    const entry = item.webkitGetAsEntry?.();
    if (entry) roots.push(entry);
  }
  const out: PathFile[] = [];
  for (const entry of roots) await walk(entry, "", out);
  return out;
}

async function walk(entry: FileSystemEntry, prefix: string, out: PathFile[]): Promise<void> {
  if (entry.isFile) {
    const file = await fileFromEntry(entry as FileSystemFileEntry);
    out.push({ file, path: `${prefix}${entry.name}` });
  } else if (entry.isDirectory) {
    const reader = (entry as FileSystemDirectoryEntry).createReader();
    const children = await readAllEntries(reader);
    for (const child of children) await walk(child, `${prefix}${entry.name}/`, out);
  }
}

function fileFromEntry(entry: FileSystemFileEntry): Promise<File> {
  return new Promise((resolve, reject) => entry.file(resolve, reject));
}

/** `readEntries` returns in batches; call repeatedly until it yields none. */
function readAllEntries(reader: FileSystemDirectoryReader): Promise<FileSystemEntry[]> {
  return new Promise((resolve, reject) => {
    const all: FileSystemEntry[] = [];
    const readBatch = (): void => {
      reader.readEntries((batch) => {
        if (batch.length === 0) {
          resolve(all);
          return;
        }
        all.push(...batch);
        readBatch();
      }, reject);
    };
    readBatch();
  });
}
