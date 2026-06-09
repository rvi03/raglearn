/**
 * `/chat` SSE route — proxies straight to the backend's `/chat` (the real serving
 * vertical) and streams its SSE through unchanged; the wire format is identical
 * (`data: {json}` with the type inside, lib/chat-types).
 *
 * Requires `FINRAG_API_URL`. There is no mock fallback — when it is unset the
 * route returns 503 so the misconfiguration is explicit.
 *
 * The frontend sends `{ message, session_id? }`; the backend expects
 * `{ text, session_id? }`. A `session_id` joins the turn to a persistent
 * conversation (short-term memory); without it the answer is stateless.
 */

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BACKEND_URL = process.env.FINRAG_API_URL;

const SSE_HEADERS = {
  "Content-Type": "text/event-stream; charset=utf-8",
  "Cache-Control": "no-cache, no-transform",
  Connection: "keep-alive",
} as const;

export async function POST(request: Request): Promise<Response> {
  if (!BACKEND_URL) {
    return new Response("backend not configured (set FINRAG_API_URL)", { status: 503 });
  }
  const { message, session_id } = (await request.json()) as {
    message?: string;
    session_id?: string;
  };

  let upstream: Response;
  try {
    upstream = await fetch(`${BACKEND_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: message ?? "", session_id: session_id ?? null }),
    });
  } catch {
    return new Response("chat backend unreachable", { status: 502 });
  }
  if (!upstream.ok || !upstream.body) {
    return new Response(`chat backend error ${upstream.status}`, { status: 502 });
  }
  return new Response(upstream.body, { headers: SSE_HEADERS });
}
