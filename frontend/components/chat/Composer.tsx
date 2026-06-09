"use client";

import { type KeyboardEvent, type ReactNode, useState } from "react";

type Props = {
  disabled: boolean;
  onSend: (text: string) => void;
};

/** Composer: bordered input + accent send square. Enter sends. */
export function Composer({ disabled, onSend }: Props): ReactNode {
  const [value, setValue] = useState("");

  const submit = (): void => {
    const text = value.trim();
    if (!text || disabled) return;
    onSend(text);
    setValue("");
  };

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>): void => {
    if (e.key === "Enter") {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="composerwrap">
      <div className="composerin">
        <div className="composer">
          <input
            placeholder="Ask a follow-up, or drop another filing to compare"
            value={value}
            disabled={disabled}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={onKeyDown}
          />
          <button
            type="button"
            className="send"
            onClick={submit}
            disabled={disabled || !value.trim()}
            aria-label="Send"
          >
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.8"
              aria-hidden="true"
            >
              <path d="M12 19V5M5 12l7-7 7 7" />
            </svg>
          </button>
        </div>
        <div className="chint">Answers cite the source document · verify against filings.</div>
      </div>
    </div>
  );
}
