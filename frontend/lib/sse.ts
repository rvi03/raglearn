import type { SSEEvent } from "@/lib/chat-types";

/**
 * Minimal SSE reader over a `fetch` Response body. Parses `data: {json}\n\n`
 * frames and yields typed events. Generic over the event type — defaults to the
 * chat `SSEEvent`; the monitor stream passes its own `MonitorEvent`. The route
 * handlers proxy the backend's SSE through unchanged — same wire format.
 */
export async function* readSSE<T = SSEEvent>(response: Response): AsyncGenerator<T> {
  if (!response.body) throw new Error("SSE response has no body");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // Frames are separated by a blank line.
    let sep = buffer.indexOf("\n\n");
    while (sep !== -1) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const event = parseFrame<T>(frame);
      if (event !== null) yield event;
      sep = buffer.indexOf("\n\n");
    }
  }
}

function parseFrame<T>(frame: string): T | null {
  const dataLines = frame
    .split("\n")
    .filter((l) => l.startsWith("data:"))
    .map((l) => l.slice(5).trimStart());
  if (dataLines.length === 0) return null;
  try {
    return JSON.parse(dataLines.join("\n")) as T;
  } catch {
    return null;
  }
}
