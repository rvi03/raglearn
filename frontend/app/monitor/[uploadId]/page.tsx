import { UploadDetail } from "@/components/monitor/UploadDetail";
import type { ReactNode } from "react";

export default async function MonitorDetailPage({
  params,
}: {
  params: Promise<{ uploadId: string }>;
}): Promise<ReactNode> {
  const { uploadId } = await params;
  return <UploadDetail uploadId={uploadId} />;
}
