import type { UploadView } from "@/lib/monitor-types";
import Link from "next/link";
import type { ReactNode } from "react";
import { MiniDag } from "./MiniDag";

/** One high-level upload row in the monitor list; click → detail trace page. */
export function UploadRow({ upload }: { upload: UploadView }): ReactNode {
  const total = upload.docs.length;
  const complete = upload.docs.every((d) => d.outcome !== null);

  return (
    <Link href={`/monitor/${upload.upload_id}`} className="urow">
      <span className={`ucbadge ${upload.country}`}>{upload.country}</span>
      <span className="urid mono">{upload.upload_id}</span>
      <span className="urfiles">
        {total} file{total > 1 ? "s" : ""}
      </span>
      <MiniDag upload={upload} />
      <span className={`urstatus ${complete ? "ok" : "run"}`}>
        {complete ? "complete" : "running"}
      </span>
      <span className="urtime" title={upload.created}>
        {upload.created?.slice(0, 16).replace("T", " ")}
      </span>
      <span className="urchev">→</span>
    </Link>
  );
}
