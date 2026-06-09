"use client";

import type { Session } from "@/lib/chat-types";
import { useCallback, useEffect, useState } from "react";

export type UseSessions = {
  sessions: Session[];
  refresh: () => Promise<void>;
  /** A new conversation id. Created lazily — it persists once its first message lands. */
  newSessionId: () => string;
  renameSession: (id: string, title: string) => Promise<void>;
  deleteSession: (id: string) => Promise<void>;
};

/**
 * The Recent rail's conversation list. Polls so a conversation appears once its
 * first message persists, and once another tab/device changes it. Rename and
 * delete update the list optimistically, then reconcile on the next poll.
 */
export function useSessions(pollMs = 5000): UseSessions {
  const [sessions, setSessions] = useState<Session[]>([]);

  const refresh = useCallback(async (): Promise<void> => {
    try {
      const res = await fetch("/api/chat/sessions", { cache: "no-store" });
      if (!res.ok) return;
      const data = (await res.json()) as { sessions?: Session[] };
      setSessions(data.sessions ?? []);
    } catch {
      // transient; keep the last good list and retry next tick
    }
  }, []);

  useEffect(() => {
    let alive = true;
    const tick = async (): Promise<void> => {
      if (alive) await refresh();
    };
    void tick();
    const id = setInterval(tick, pollMs);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [pollMs, refresh]);

  const newSessionId = useCallback((): string => crypto.randomUUID(), []);

  const renameSession = useCallback(async (id: string, title: string): Promise<void> => {
    setSessions((prev) => prev.map((s) => (s.id === id ? { ...s, title } : s)));
    try {
      await fetch(`/api/chat/sessions/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      });
    } catch {
      // the next poll restores the server's truth if this failed
    }
  }, []);

  const deleteSession = useCallback(async (id: string): Promise<void> => {
    setSessions((prev) => prev.filter((s) => s.id !== id));
    try {
      await fetch(`/api/chat/sessions/${id}`, { method: "DELETE" });
    } catch {
      // the next poll restores the server's truth if this failed
    }
  }, []);

  return { sessions, refresh, newSessionId, renameSession, deleteSession };
}
