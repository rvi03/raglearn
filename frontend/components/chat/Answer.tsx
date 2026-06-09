"use client";

import type { Citation } from "@/lib/chat-types";
import { Fragment, type ReactNode } from "react";

type Props = {
  text: string;
  citations: Citation[];
  activeCite: number | null;
  streaming: boolean;
  onCite: (id: number) => void;
};

type Ctx = {
  valid: Set<number>;
  activeCite: number | null;
  onCite: (id: number) => void;
};

// One pass over inline text: a bold span, a `[n]` citation, or a figure
// (currency / percent / magnitude / signed delta). Everything else is plain text.
const INLINE =
  /(\*\*[^*]+?\*\*|\[\d+\]|[+\-−–]\s?\d[\d,]*\.?\d*\s?%|[$₹€£]\s?\d[\d,]*\.?\d*\s?(?:billion|bn|million|mn|crore|cr|trillion|tn|lakh|k)?|\d[\d,]*\.?\d*\s?%|\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d[\d,]*\.?\d*\s?(?:billion|bn|million|mn|crore|cr|trillion|tn|lakh))/gi;

const LIST_MARKER = /^\s*([-*•]|\d+[.)])\s+/;

/** Figure styling: signed % deltas carry direction colour, other figures stay neutral mono. */
function figureClass(token: string): string {
  const head = token.trimStart()[0];
  if (head === "+") return "num up";
  if (head === "-" || head === "−" || head === "–") return "num down";
  return "num";
}

/** Render inline text: bold, citation chips, and styled figures. */
function renderInline(text: string, ctx: Ctx, key: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  let last = 0;
  let n = 0;
  INLINE.lastIndex = 0;
  let m: RegExpExecArray | null = INLINE.exec(text);
  while (m !== null) {
    if (m.index > last) {
      nodes.push(<Fragment key={`${key}-t${n++}`}>{text.slice(last, m.index)}</Fragment>);
    }
    const tok = m[0];
    if (tok.startsWith("**")) {
      nodes.push(
        <strong key={`${key}-b${n++}`}>
          {renderInline(tok.slice(2, -2), ctx, `${key}-b${n}`)}
        </strong>,
      );
    } else if (/^\[\d+\]$/.test(tok)) {
      const id = Number(tok.slice(1, -1));
      nodes.push(
        ctx.valid.has(id) ? (
          <button
            key={`${key}-c${n++}`}
            type="button"
            className={`cite${ctx.activeCite === id ? " active" : ""}`}
            onClick={() => ctx.onCite(id)}
            aria-label={`Citation ${id}`}
          >
            {id}
          </button>
        ) : (
          <Fragment key={`${key}-c${n++}`}>{tok}</Fragment>
        ),
      );
    } else {
      nodes.push(
        <span key={`${key}-n${n++}`} className={figureClass(tok)}>
          {tok}
        </span>,
      );
    }
    last = m.index + tok.length;
    m = INLINE.exec(text);
  }
  if (last < text.length) nodes.push(<Fragment key={`${key}-e`}>{text.slice(last)}</Fragment>);
  return nodes;
}

/** Answer body: renders markdown blocks (paragraphs + bullet/numbered lists) with
 * inline bold, citation chips, and figure styling. */
export function Answer({ text, citations, activeCite, streaming, onCite }: Props): ReactNode {
  const ctx: Ctx = { valid: new Set(citations.map((c) => c.id)), activeCite, onCite };
  const blocks = text.split(/\n{2,}/);

  return (
    <div className="amsg">
      {blocks.map((block, bi) => {
        const lines = block.split("\n").filter((l) => l.trim() !== "");
        if (lines.length === 0) return null;
        const isLast = bi === blocks.length - 1;

        if (lines.every((l) => LIST_MARKER.test(l))) {
          const ordered = /^\s*\d+[.)]/.test(lines[0]);
          const items = lines.map((l) => l.replace(LIST_MARKER, ""));
          const body = items.map((it, ii) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: render-only, stable within a static answer
            <li key={ii}>{renderInline(it, ctx, `${bi}-${ii}`)}</li>
          ));
          return ordered ? (
            // biome-ignore lint/suspicious/noArrayIndexKey: render-only, no reorder
            <ol key={bi} className="alist">
              {body}
            </ol>
          ) : (
            // biome-ignore lint/suspicious/noArrayIndexKey: render-only, no reorder
            <ul key={bi} className="alist">
              {body}
            </ul>
          );
        }

        return (
          // biome-ignore lint/suspicious/noArrayIndexKey: render-only, no reorder
          <p key={bi}>
            {renderInline(lines.join(" "), ctx, `${bi}`)}
            {streaming && isLast && <span className="caret" aria-hidden="true" />}
          </p>
        );
      })}
    </div>
  );
}
