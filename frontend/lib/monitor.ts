import type { MacroNode, MacroStatus, NodeStatus, UploadView } from "@/lib/monitor-types";
import { MACRO_STAGES, NODE_TO_MACRO } from "@/lib/monitor-types";

/** Reduce one macro stage's node statuses (across all docs) to a single status. */
function aggregate(statuses: NodeStatus[], complete: boolean): MacroStatus {
  if (statuses.length === 0) return complete ? "na" : "pending";
  if (statuses.includes("running")) return "running";
  if (statuses.includes("failed")) return "failed";
  if (statuses.includes("done")) return "done"; // at least one done (others may be skip/defer)
  if (statuses.includes("deferred")) return "deferred";
  return "skipped";
}

/**
 * Collapse an upload's detailed per-doc traces into the high-level macro flow
 * (detect → parse → chunk → embed → index), one aggregate status per stage.
 */
export function computeMacroFlow(upload: UploadView): MacroNode[] {
  const complete = upload.docs.every((d) => d.outcome !== null);
  const buckets: Record<string, NodeStatus[]> = {};
  for (const doc of upload.docs) {
    for (const node of doc.trace) {
      const macro = NODE_TO_MACRO[node.id];
      if (!macro) continue;
      if (!buckets[macro]) buckets[macro] = [];
      buckets[macro].push(node.status);
    }
  }
  return MACRO_STAGES.map((stage) => ({
    id: stage.id,
    label: stage.label,
    status: aggregate(buckets[stage.id] ?? [], complete),
  }));
}
