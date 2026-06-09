"use client";

import { useChat } from "@/hooks/useChat";
import { EXAMPLE_PROMPTS, FOLLOW_UPS } from "@/lib/chat-prompts";
import type { Citation, Source } from "@/lib/chat-types";
import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { Composer } from "./Composer";
import { SourceViewer } from "./SourceViewer";
import { Turn } from "./Turn";

type ViewerState = { source: Source; citation: Citation | null };

/** The Chat view: context header → streamed thread → composer → source drawer. */
export function ChatView({ sessionId }: { sessionId: string }): ReactNode {
  const { messages, isStreaming, sendMessage } = useChat(sessionId);
  const [viewer, setViewer] = useState<ViewerState | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // biome-ignore lint/correctness/useExhaustiveDependencies: scroll on every message change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const openFor = useMemo(
    () =>
      (sources: Source[], citations: Citation[]) =>
      (id: number): void => {
        const source = sources.find((s) => s.id === id);
        if (!source) return;
        setViewer({ source, citation: citations.find((c) => c.id === id) ?? null });
      },
    [],
  );

  const isEmpty = messages.length === 0;
  const lastId = messages.length > 0 ? messages[messages.length - 1].id : null;

  return (
    <>
      <div className="mhead">
        <div className="ctx">
          <div className="fic">RAG</div>
          <div className="nm">
            Financial filings
            <small>ask · grounded · cited</small>
          </div>
        </div>
        <div className="mtools">
          <div className="mode">
            <span className="d" />
            Agentic · RAG
          </div>
        </div>
      </div>

      <div className="scroll">
        {isEmpty ? (
          <div className="empty">
            <div className="ekick">finrag · assistant</div>
            <h2>Ask about a filing</h2>
            <p>Answers are grounded in indexed passages, with inline citations you can open.</p>
            <div className="examples">
              {EXAMPLE_PROMPTS.map((p, i) => (
                <button key={p} type="button" className="example" onClick={() => sendMessage(p)}>
                  <span className="en">{String(i + 1).padStart(2, "0")}</span>
                  {p}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="thread">
            {messages.map((m) => (
              <Turn
                key={m.id}
                message={m}
                activeCite={viewer?.citation?.id ?? null}
                isLast={m.id === lastId && m.role === "assistant"}
                followUps={FOLLOW_UPS}
                onOpen={openFor(m.sources, m.citations)}
                onFollow={sendMessage}
              />
            ))}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      <Composer disabled={isStreaming} onSend={sendMessage} />

      {viewer && (
        <SourceViewer
          source={viewer.source}
          citation={viewer.citation}
          onClose={() => setViewer(null)}
        />
      )}
    </>
  );
}
