import "./finrag.css";
import { AppShell } from "@/components/AppShell";
import { ThemeProvider } from "@/components/ThemeProvider";
import type { Metadata } from "next";
import { IBM_Plex_Mono, Space_Grotesk } from "next/font/google";
import type { ReactNode } from "react";

const grotesk = Space_Grotesk({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-grotesk",
  display: "swap",
});

const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "finrag",
  description: "Agentic RAG over financial filings",
};

export default function RootLayout({
  children,
}: {
  children: ReactNode;
}): ReactNode {
  return (
    <html lang="en" className={`${grotesk.variable} ${mono.variable}`}>
      <body>
        <ThemeProvider>
          <AppShell>{children}</AppShell>
        </ThemeProvider>
      </body>
    </html>
  );
}
