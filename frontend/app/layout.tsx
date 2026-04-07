import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'ResearchRAG',
  description: 'Research paper understanding and content generation',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      {/*
        suppressHydrationWarning on <body> silences hydration mismatches
        caused by browser extensions (password managers, form autofill tools)
        that inject attributes like fdprocessedid onto form elements before
        React hydrates. This is safe — it only suppresses warnings on the
        body element itself, not on child components.
      */}
      <body suppressHydrationWarning>{children}</body>
    </html>
  )
}