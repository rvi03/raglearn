/**
 * `/api/ingestion/uploads` — the persistent corpus list. Proxies the backend's
 * `GET /ingestion/uploads` (durable per-document ingestion status from Postgres).
 *
 * Requires `FINRAG_API_URL`; with no backend it returns an empty list so the
 * corpus view renders cleanly rather than erroring.
 */

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BACKEND_URL = process.env.FINRAG_API_URL;

export async function GET(): Promise<Response> {
  if (!BACKEND_URL) {
    return Response.json({ uploads: [] });
  }
  try {
    const upstream = await fetch(`${BACKEND_URL}/ingestion/uploads`, { cache: "no-store" });
    if (!upstream.ok) {
      return new Response(`uploads backend error ${upstream.status}`, { status: 502 });
    }
    return new Response(upstream.body, {
      headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
    });
  } catch {
    return new Response("uploads backend unreachable", { status: 502 });
  }
}
