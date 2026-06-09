/**
 * `/api/monitor/stream` SSE route — the live ingestion DAG feed. Proxies straight
 * to the backend's `/ingestion/events` (the Redis-backed stage emitter bridged to
 * SSE) and streams it through unchanged; the wire format is identical
 * (`data: {json}` with the type inside, lib/monitor-types).
 *
 * Requires `FINRAG_API_URL`. There is no mock fallback — when it is unset the
 * route returns 503 so the misconfiguration is explicit.
 */

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BACKEND_URL = process.env.FINRAG_API_URL;

const SSE_HEADERS = {
  "Content-Type": "text/event-stream; charset=utf-8",
  "Cache-Control": "no-cache, no-transform",
  Connection: "keep-alive",
} as const;

export async function GET(): Promise<Response> {
  if (!BACKEND_URL) {
    return new Response("backend not configured (set FINRAG_API_URL)", { status: 503 });
  }

  let upstream: Response;
  try {
    upstream = await fetch(`${BACKEND_URL}/ingestion/events`, {
      headers: { Accept: "text/event-stream" },
    });
  } catch {
    return new Response("monitor backend unreachable", { status: 502 });
  }
  if (!upstream.ok || !upstream.body) {
    return new Response(`monitor backend error ${upstream.status}`, { status: 502 });
  }
  return new Response(upstream.body, { headers: SSE_HEADERS });
}
