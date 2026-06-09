"use client";

import type { Citation, Source } from "@/lib/chat-types";
import { type ReactNode, useState } from "react";

type Props = {
  sources: Source[];
  citations: Citation[];
  onOpen: (id: number) => void;
};

/** Collapsible in-thread evidence rows ("Analyzed N passages"). */
export function SourcesList({ sources, citations, onOpen }: Props): ReactNode {
  const [open, setOpen] = useState(false);
  if (sources.length === 0) return null;

  const pageOf = (id: number): number | null => citations.find((c) => c.id === id)?.page ?? null;

  return (
    <div className={`srcwrap${open ? " open" : ""}`}>
      <button type="button" className="srctoggle" onClick={() => setOpen((v) => !v)}>
        Analyzed {sources.length} passages
        <span className="car">▾</span>
      </button>
      <div className="srclist">
        {sources.map((s) => {
          const page = pageOf(s.id);
          return (
            <button key={s.id} type="button" className="srcrow" onClick={() => onOpen(s.id)}>
              <span className="sn">[{s.id}]</span>
              <span>
                <span className="stitle">{s.title}</span>
                <span className="ssnip">{s.snippet}</span>
              </span>
              <span className="spg">{page !== null ? `p.${page}` : ""}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
