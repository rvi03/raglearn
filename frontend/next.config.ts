import type { NextConfig } from "next";

// Backend API base; same-origin proxy below avoids CORS in dev.
const API_URL = process.env.FINRAG_API_URL ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Hide the floating dev-tools indicator (dev-only; never in prod builds).
  devIndicators: false,
  experimental: {
    // A filing folder of PDFs (e.g. an India results pack) easily exceeds Next's
    // default request-body cap; the proxy below buffers the multipart body, so a
    // larger upload 500s in the proxy before it ever reaches the backend. Raise the
    // cap to fit real filing folders. Buffered in memory — fine at dev/filing scale.
    middlewareClientMaxBodySize: "100mb",
  },
  // Proxy backend routes so the browser calls same-origin (no CORS).
  async rewrites() {
    return [
      { source: "/ingest/:path*", destination: `${API_URL}/ingest/:path*` },
      // "Open full document" links point at /sources/<doc_id>; proxy to the backend.
      { source: "/sources/:path*", destination: `${API_URL}/sources/:path*` },
    ];
  },
};

export default nextConfig;
