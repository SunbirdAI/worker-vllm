import type { NextConfig } from "next";

const isExport = process.env.BUILD_MODE === "export";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  images: { unoptimized: true },
  ...(isExport ? { output: "export" as const } : {}),
  async rewrites() {
    if (isExport) return [];
    // Local dev: proxy API calls to the FastAPI backend on :8000.
    const backend = process.env.BACKEND_URL ?? "http://localhost:8000";
    return [
      { source: "/health", destination: `${backend}/health` },
      { source: "/generate", destination: `${backend}/generate` },
      { source: "/generate_stream", destination: `${backend}/generate_stream` },
      { source: "/generate_production", destination: `${backend}/generate_production` },
    ];
  },
};

export default nextConfig;
