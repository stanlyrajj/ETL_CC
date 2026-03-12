import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ResearchRAG",
  description: "Research papers, taught conversationally.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}