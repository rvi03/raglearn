import type { DocView, NodeStatus } from "@/lib/monitor-types";
import type { ReactNode } from "react";

const GLYPH: Record<NodeStatus, string> = {
  running: "◐",
  done: "●",
  skipped: "–",
  deferred: "⏸",
  failed: "✕",
};

/** Vertical execution trace for one document — node label + branch decision. */
export function DocTrace({ doc }: { doc: DocView }): ReactNode {
  return (
    <div className="trace2">
      {doc.trace.map((node) => (
        <div key={node.id} className={`tnode st-${node.status}`}>
          <span className="g">{GLYPH[node.status]}</span>
          <span className="tlabel">{node.label}</span>
          {node.detail && <span className="tarrow">→</span>}
          {node.detail && <span className="tdetail">{node.detail}</span>}
        </div>
      ))}
    </div>
  );
}
