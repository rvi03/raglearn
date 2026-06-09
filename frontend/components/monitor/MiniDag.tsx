import { computeMacroFlow } from "@/lib/monitor";
import type { MacroStatus, UploadView } from "@/lib/monitor-types";
import { Fragment, type ReactNode } from "react";

const GLYPH: Record<MacroStatus, string> = {
  done: "●",
  running: "◐",
  pending: "○",
  deferred: "⏸",
  skipped: "–",
  na: "·",
  failed: "✕",
};

/** Small high-level pipeline flow for one upload (detect → … → index). */
export function MiniDag({ upload }: { upload: UploadView }): ReactNode {
  const flow = computeMacroFlow(upload);
  return (
    <span className="minidag">
      {flow.map((node, i) => (
        <Fragment key={node.id}>
          {i > 0 && <span className="msep">→</span>}
          <span className={`mnode st-${node.status}`} title={`${node.label}: ${node.status}`}>
            <span className="g">{GLYPH[node.status]}</span>
            {node.label}
          </span>
        </Fragment>
      ))}
    </span>
  );
}
