/**
 * `/api/chat/sessions/[sessionId]` — rename and delete one conversation. Proxies
 * the backend's `PATCH/DELETE /chat/sessions/{id}` (404 passes through when the
 * conversation does not exist).
 */

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BACKEND_URL = process.env.FINRAG_API_URL;

type Params = { params: Promise<{ sessionId: string }> };

export async function PATCH(request: Request, { params }: Params): Promise<Response> {
  if (!BACKEND_URL) {
    return new Response("backend not configured (set FINRAG_API_URL)", { status: 503 });
  }
  const { sessionId } = await params;
  const body = await request.text();
  try {
    const upstream = await fetch(`${BACKEND_URL}/chat/sessions/${sessionId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body,
    });
    return new Response(upstream.body, {
      status: upstream.status,
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    return new Response("sessions backend unreachable", { status: 502 });
  }
}

export async function DELETE(_request: Request, { params }: Params): Promise<Response> {
  if (!BACKEND_URL) {
    return new Response("backend not configured (set FINRAG_API_URL)", { status: 503 });
  }
  const { sessionId } = await params;
  try {
    const upstream = await fetch(`${BACKEND_URL}/chat/sessions/${sessionId}`, { method: "DELETE" });
    return new Response(upstream.body, {
      status: upstream.status,
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    return new Response("sessions backend unreachable", { status: 502 });
  }
}
