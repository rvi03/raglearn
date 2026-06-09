"use client";

import type { ChatMessage } from "@/lib/chat-types";
import type { ReactNode } from "react";
import { AgentTrace } from "./AgentTrace";
import { Answer } from "./Answer";
import { GroundingBadge } from "./GroundingBadge";
import { SafetyNote } from "./SafetyNote";
import { SourcesList } from "./SourcesList";

type Props = {
  message: ChatMessage;
  activeCite: number | null;
  onOpen: (id: number) => void;
};

/** One thread turn — user prompt or the assembled assistant answer. */
export function Turn({ message, activeCite, onOpen }: Props): ReactNode {
  if (message.role === "user") {
    return (
      <div className="turn">
        <div className="role me">You</div>
        <div className="umsg">{message.text}</div>
      </div>
    );
  }

  const done = !message.streaming;

  return (
    <div className="turn">
      <div className="role">Assistant</div>
      <AgentTrace steps={message.steps} costUsd={message.costUsd} />
      <Answer
        text={message.text}
        citations={message.citations}
        activeCite={activeCite}
        streaming={message.streaming}
        onCite={onOpen}
      />
      {done && (
        <>
          <SafetyNote answered={message.answered} redacted={message.redacted} />
          <GroundingBadge confidence={message.groundingConfidence} />
          <SourcesList sources={message.sources} citations={message.citations} onOpen={onOpen} />
        </>
      )}
    </div>
  );
}
