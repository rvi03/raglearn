"use client";

import { useSessions } from "@/hooks/useSessions";
import { useRouter } from "next/navigation";
import type { ReactNode } from "react";

/** Chats — the full conversation history. (The rail shows only the recent five.) */
export default function ChatsPage(): ReactNode {
  const router = useRouter();
  const { sessions } = useSessions();

  return (
    <>
      <div className="mhead">
        <div className="ctx">
          <div className="fic">CHAT</div>
          <div className="nm">
            Chats
            <small>your conversation history</small>
          </div>
        </div>
      </div>

      <div className="scroll">
        {sessions.length === 0 ? (
          <div className="empty">
            <div className="ekick">finrag · chats</div>
            <h2>No chats yet</h2>
            <p>Start one with “Ask”. Your conversations will appear here.</p>
          </div>
        ) : (
          <div className="chatlist">
            {sessions.map((s) => (
              <button
                key={s.id}
                type="button"
                className="chatrow"
                onClick={() => router.push(`/chat/${s.id}`)}
              >
                <span className="chatrow-title">{s.title}</span>
                <span className="chatrow-time">{formatWhen(s.updated_at)}</span>
              </button>
            ))}
          </div>
        )}
      </div>
    </>
  );
}

/** Short, locale-aware date for a conversation's last activity. */
function formatWhen(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}
