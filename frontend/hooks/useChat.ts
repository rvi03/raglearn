"use client";

import type { ChatMessage, Citation, SSEEvent, Source, StoredMessage } from "@/lib/chat-types";
import { readSSE } from "@/lib/sse";
import { useCallback, useEffect, useRef, useState } from "react";

/** Rebuild a thread message from its persisted form when a conversation reopens. */
function fromStored(m: StoredMessage): ChatMessage {
  const meta = (m.meta ?? {}) as Record<string, unknown>;
  return {
    id: m.id,
    role: m.role,
    text: m.text,
    citations: (meta.citations as Citation[] | undefined) ?? [],
    sources: (meta.sources as Source[] | undefined) ?? [],
    steps: [], // the live agent trace is not persisted; reopened turns show none
    costUsd: (meta.costUsd as number | undefined) ?? null,
    traceId: (meta.traceId as string | undefined) ?? null,
    groundingConfidence: (meta.groundingConfidence as number | undefined) ?? null,
    answered: (meta.answered as boolean | undefined) ?? true,
    redacted: (meta.redacted as string[] | undefined) ?? [],
    streaming: false,
  };
}

function emptyAssistant(id: string): ChatMessage {
  return {
    id,
    role: "assistant",
    text: "",
    citations: [],
    sources: [],
    steps: [],
    costUsd: null,
    traceId: null,
    groundingConfidence: null,
    answered: true,
    redacted: [],
    streaming: true,
  };
}

/** Apply one SSE event to the in-flight assistant message. */
function reduceEvent(msg: ChatMessage, event: SSEEvent): ChatMessage {
  switch (event.type) {
    case "token":
      return { ...msg, text: msg.text + event.delta };
    case "source":
      return {
        ...msg,
        sources: [
          ...msg.sources,
          { id: event.id, title: event.title, url: event.url, snippet: event.snippet },
        ],
      };
    case "citation":
      return {
        ...msg,
        citations: [
          ...msg.citations,
          { id: event.id, source_doc_id: event.source_doc_id, page: event.page, span: event.span },
        ],
      };
    case "agent_step":
      return {
        ...msg,
        steps: [
          ...msg.steps,
          {
            name: event.name,
            path: event.path,
            detail: event.detail ?? "",
            status: event.status,
            latency_ms: event.latency_ms,
            tokens_in: event.tokens_in,
            tokens_out: event.tokens_out,
            model: event.model,
            usd: event.usd,
          },
        ],
      };
    case "done":
      return {
        ...msg,
        costUsd: event.query_usd,
        traceId: event.trace_id,
        groundingConfidence: event.grounding_confidence ?? null,
        answered: event.answered ?? true,
        redacted: event.redacted ?? [],
        streaming: false,
      };
    default:
      return msg;
  }
}

export type UseChat = {
  messages: ChatMessage[];
  isStreaming: boolean;
  sendMessage: (text: string) => Promise<void>;
};

/**
 * Drive one conversation: load its persisted transcript on open, then stream new
 * turns. The turn is sent with ``sessionId`` so the backend persists it and folds
 * the recent turns back in as short-term memory.
 */
export function useChat(sessionId: string): UseChat {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const streamingRef = useRef(false);

  // Reopen the conversation: replace the thread with its stored transcript when
  // the session changes. A fresh conversation simply loads empty.
  useEffect(() => {
    let alive = true;
    setMessages([]);
    void (async () => {
      try {
        const res = await fetch(`/api/chat/sessions/${sessionId}/messages`, { cache: "no-store" });
        if (!res.ok) return;
        const data = (await res.json()) as { messages?: StoredMessage[] };
        if (alive && data.messages) setMessages(data.messages.map(fromStored));
      } catch {
        // a fresh/unreachable conversation just stays empty
      }
    })();
    return () => {
      alive = false;
    };
  }, [sessionId]);

  const sendMessage = useCallback(
    async (text: string): Promise<void> => {
      const trimmed = text.trim();
      if (!trimmed || streamingRef.current) return;

      streamingRef.current = true;
      setIsStreaming(true);

      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        text: trimmed,
        citations: [],
        sources: [],
        steps: [],
        costUsd: null,
        traceId: null,
        groundingConfidence: null,
        answered: true,
        redacted: [],
        streaming: false,
      };
      const assistantId = crypto.randomUUID();
      setMessages((prev) => [...prev, userMsg, emptyAssistant(assistantId)]);

      const patch = (event: SSEEvent): void => {
        setMessages((prev) => prev.map((m) => (m.id === assistantId ? reduceEvent(m, event) : m)));
      };

      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: trimmed, session_id: sessionId }),
        });
        if (!res.ok) throw new Error(`chat request failed: ${res.status}`);
        for await (const event of readSSE(res)) patch(event);
      } catch {
        patch({
          type: "token",
          delta: "\n\n_Could not reach the chat service._",
        });
        patch({
          type: "done",
          usage: { tokens_in: 0, tokens_out: 0 },
          trace_id: "error",
          query_usd: 0,
        });
      } finally {
        // Ensure the turn is never left visually streaming.
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? { ...m, streaming: false } : m)),
        );
        streamingRef.current = false;
        setIsStreaming(false);
      }
    },
    [sessionId],
  );

  return { messages, isStreaming, sendMessage };
}
