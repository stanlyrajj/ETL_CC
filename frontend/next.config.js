/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      // SSE endpoints must be excluded from the Next.js proxy rewrite.
      // Next.js dev server buffers streamed responses, so SSE events never
      // reach the browser through the proxy. These paths connect directly
      // to the backend instead (see NEXT_PUBLIC_BACKEND_URL in api.ts).
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
    ]
  },
}

module.exports = nextConfig