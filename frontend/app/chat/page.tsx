"use client";

import { useRouter } from "next/navigation";
import { type ReactNode, useEffect } from "react";

/**
 * `/chat` opens a fresh conversation: mint a new id and redirect to its route.
 * The conversation persists (and joins Recent) once its first message lands, so
 * abandoning a blank thread leaves nothing behind.
 */
export default function ChatPage(): ReactNode {
  const router = useRouter();
  useEffect(() => {
    router.replace(`/chat/${crypto.randomUUID()}`);
  }, [router]);
  return null;
}
