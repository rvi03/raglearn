import type { ReactNode } from "react";

/**
 * Settings — placeholder only. Surfaces the idea of swapping pipeline models
 * (embedder / LLM / reranker) from the UI; NOT wired to the backend. The real
 * control will drive the `/config` adapter matrix when built.
 */

type ModelRow = { id: string; label: string; options: string[] };

const MODEL_ROWS: ModelRow[] = [
  {
    id: "embedding",
    label: "Embedding model",
    options: ["bge-m3", "e5-large-v2", "nomic-embed-text", "Fin-E5"],
  },
  {
    id: "llm",
    label: "LLM (generation)",
    options: ["qwen2.5-7b-instruct", "qwen2.5-14b-instruct", "llama-3.1-8b", "gpt-4o (cloud)"],
  },
  {
    id: "reranker",
    label: "Reranker",
    options: ["bge-reranker-v2-m3", "jina-reranker-v2", "none"],
  },
];

export default function SettingsPage(): ReactNode {
  return (
    <div className="placeholder">
      <h1>Settings</h1>
      <p>
        Swap pipeline models from the UI. <span className="setnote">placeholder</span>
      </p>
      <div className="setgrid">
        {MODEL_ROWS.map((row) => (
          <div key={row.id} className="setrow">
            <span className="lbl">{row.label}</span>
            {/* Non-functional placeholder — selection is not persisted or applied yet. */}
            <select defaultValue={row.options[0]} aria-label={row.label}>
              {row.options.map((opt) => (
                <option key={opt} value={opt}>
                  {opt}
                </option>
              ))}
            </select>
          </div>
        ))}
      </div>
    </div>
  );
}
