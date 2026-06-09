/**
 * `/api/chat/sessions/[sessionId]/messages` — a conversation's full transcript,
 * used to reopen it. Proxies the backend's `GET /chat/sessions/{id}/messages`.
 *
 * With no backend configured it returns an empty transcript so a fresh
 * conversation renders cleanly.
 */

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BACKEND_URL = process.env.FINRAG_API_URL;

type Params = { params: Promise<{ sessionId: string }> };

export async function GET(_request: Request, { params }: Params): Promise<Response> {
  if (!BACKEND_URL) {
    return Response.json({ messages: [] });
  }
  const { sessionId } = await params;
  try {
    const upstream = await fetch(`${BACKEND_URL}/chat/sessions/${sessionId}/messages`, {
      cache: "no-store",
    });
    if (!upstream.ok) {
      return new Response(`messages backend error ${upstream.status}`, { status: 502 });
    }
    return new Response(upstream.body, {
      headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
    });
  } catch {
    return new Response("messages backend unreachable", { status: 502 });
  }
}
