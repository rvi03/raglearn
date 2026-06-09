"use client";

import type { AgentStep } from "@/lib/chat-types";
import { type ReactNode, useState } from "react";

type Props = {
  steps: AgentStep[];
  costUsd: number | null;
};

const fmtUsd = (n: number): string => `$${n.toFixed(4)}`;

/** Single expandable agent-trace line with an accent token + per-step detail. */
export function AgentTrace({ steps, costUsd }: Props): ReactNode {
  const [open, setOpen] = useState(false);
  if (steps.length === 0) return null;

  const totalMs = steps.reduce((a, s) => a + s.latency_ms, 0);

  return (
    <div className={`trace${open ? " open" : ""}`}>
      <button type="button" className="traceline" onClick={() => setOpen((v) => !v)}>
        <span className="tk">λ</span>
        <span>
          {steps.length} steps · {totalMs}ms{costUsd !== null ? ` · ${fmtUsd(costUsd)}` : ""}
        </span>
        <span className="car">▾</span>
      </button>
      <div className="tsteps">
        {steps.map((s, i) => (
          <div key={`${s.name}-${s.path}-${i}`} className="tstep">
            <span className="tstepl">
              <b>{s.name}</b>
              {s.path ? <span className="tpath"> {s.path}</span> : null}
              {s.detail ? <span className="tdetail"> · {s.detail}</span> : null}
            </span>
            <span className="tnums">
              {s.tokens_in + s.tokens_out > 0 ? `${s.tokens_in + s.tokens_out} tok · ` : ""}
              {s.latency_ms}ms · <span className="u">{fmtUsd(s.usd)}</span>
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
