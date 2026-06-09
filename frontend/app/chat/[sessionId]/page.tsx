import { ChatView } from "@/components/chat/ChatView";
import type { ReactNode } from "react";

/** A single conversation, addressed by its id (shareable, back/forward navigable). */
export default async function ChatSessionPage({
  params,
}: {
  params: Promise<{ sessionId: string }>;
}): Promise<ReactNode> {
  const { sessionId } = await params;
  return <ChatView sessionId={sessionId} />;
}
