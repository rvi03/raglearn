/**
 * `/api/chat/sessions` — list and create conversations. Proxies the backend's
 * `GET/POST /chat/sessions` (durable conversations in Postgres).
 *
 * With no backend configured, GET returns an empty list so the rail renders
 * cleanly, and POST returns 503 so a misconfiguration is explicit.
 */

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BACKEND_URL = process.env.FINRAG_API_URL;

export async function GET(): Promise<Response> {
  if (!BACKEND_URL) {
    return Response.json({ sessions: [] });
  }
  try {
    const upstream = await fetch(`${BACKEND_URL}/chat/sessions`, { cache: "no-store" });
    if (!upstream.ok) {
      return new Response(`sessions backend error ${upstream.status}`, { status: 502 });
    }
    return new Response(upstream.body, {
      headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
    });
  } catch {
    return new Response("sessions backend unreachable", { status: 502 });
  }
}

export async function POST(request: Request): Promise<Response> {
  if (!BACKEND_URL) {
    return new Response("backend not configured (set FINRAG_API_URL)", { status: 503 });
  }
  const body = await request.text();
  try {
    const upstream = await fetch(`${BACKEND_URL}/chat/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body || "{}",
    });
    return new Response(upstream.body, {
      status: upstream.status,
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    return new Response("sessions backend unreachable", { status: 502 });
  }
}
