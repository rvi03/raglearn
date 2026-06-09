"use client";

import { useTheme } from "@/components/ThemeProvider";
import { useSessions } from "@/hooks/useSessions";
import type { Session } from "@/lib/chat-types";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { type ReactNode, useState } from "react";

/** One conversation in the Recent rail: open on click, rename inline, delete. */
function ConvItem({
  session,
  active,
  onOpen,
  onRename,
  onDelete,
}: {
  session: Session;
  active: boolean;
  onOpen: () => void;
  onRename: (title: string) => void;
  onDelete: () => void;
}): ReactNode {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(session.title);

  const commit = (): void => {
    const title = draft.trim();
    if (title && title !== session.title) onRename(title);
    setEditing(false);
  };

  if (editing) {
    return (
      <input
        className="convedit"
        // biome-ignore lint/a11y/noAutofocus: focus the field the user just opened
        autoFocus
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") commit();
          if (e.key === "Escape") {
            setDraft(session.title);
            setEditing(false);
          }
        }}
      />
    );
  }

  return (
    <div className={active ? "conv on" : "conv"}>
      <button
        type="button"
        className="convtitle"
        onClick={onOpen}
        onDoubleClick={() => {
          setDraft(session.title);
          setEditing(true);
        }}
      >
        {session.title}
      </button>
      <button type="button" className="convdel" aria-label="Delete conversation" onClick={onDelete}>
        ×
      </button>
    </div>
  );
}

/** Custom rail + main frame. */
export function AppShell({ children }: { children: ReactNode }): ReactNode {
  const pathname = usePathname();
  const router = useRouter();
  const { theme, toggle } = useTheme();
  const { sessions, newSessionId, renameSession, deleteSession } = useSessions();
  // "Chats" (history) covers both the full-history page and any open thread.
  const isChats = pathname.startsWith("/chats") || pathname.startsWith("/chat/");
  const isDocs = pathname.startsWith("/documents");
  const isMonitor = pathname.startsWith("/monitor");
  const isSettings = pathname.startsWith("/settings");

  // Start a fresh conversation: a new id routes to a blank thread; it persists
  // (and joins Recent) once its first message lands.
  const newChat = (): void => router.push(`/chat/${newSessionId()}`);

  const removeConversation = (id: string): void => {
    void deleteSession(id);
    if (pathname === `/chat/${id}`) newChat(); // don't strand the user on a deleted thread
  };

  return (
    <div className="app">
      <aside className="rail">
        <div className="brand">
          <div className="glyph">r</div>
          <b>finrag</b>
        </div>

        <nav className="nav">
          <Link href="/documents" className={isDocs ? "on" : undefined}>
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.7"
              aria-hidden="true"
            >
              <path d="M6 3h9l4 4v14H6zM15 3v4h4" />
            </svg>
            Documents
          </Link>
          <Link href="/monitor" className={isMonitor ? "on" : undefined}>
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.7"
              aria-hidden="true"
            >
              <path d="M3 12h4l2 6 4-14 2 8h6" />
            </svg>
            Monitor
          </Link>
          <Link href="/chats" className={isChats ? "on" : undefined}>
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.7"
              aria-hidden="true"
            >
              <path d="M4 4h16v12H8l-4 4zM9 9h6M9 12h4" />
            </svg>
            Chats
          </Link>
        </nav>

        <button type="button" className="newbtn" onClick={newChat}>
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            aria-hidden="true"
          >
            <path d="M12 5v14M5 12h14" />
          </svg>
          Ask
        </button>

        <div className="railsec">
          <span className="lbl">Recent</span>
        </div>
        <div className="convs">
          {sessions.length === 0 ? (
            <div className="convempty">No chats yet</div>
          ) : (
            sessions
              .slice(0, 5)
              .map((s) => (
                <ConvItem
                  key={s.id}
                  session={s}
                  active={pathname === `/chat/${s.id}`}
                  onOpen={() => router.push(`/chat/${s.id}`)}
                  onRename={(title) => void renameSession(s.id, title)}
                  onDelete={() => removeConversation(s.id)}
                />
              ))
          )}
        </div>

        <div className="railctl">
          <button type="button" className="ctlbtn" onClick={toggle}>
            {theme === "dark" ? (
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.7"
                aria-hidden="true"
              >
                <circle cx="12" cy="12" r="4" />
                <path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5L19 19M19 5l-1.5 1.5M6.5 17.5L5 19" />
              </svg>
            ) : (
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.7"
                aria-hidden="true"
              >
                <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
              </svg>
            )}
            {theme === "dark" ? "Light theme" : "Dark theme"}
          </button>
          <Link href="/settings" className={isSettings ? "ctlbtn on" : "ctlbtn"}>
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.7"
              aria-hidden="true"
            >
              <path d="M4 7h9M19 7h1M4 17h1M10 17h10" />
              <circle cx="16" cy="7" r="2.4" />
              <circle cx="7" cy="17" r="2.4" />
            </svg>
            Settings
          </Link>
        </div>
      </aside>

      <section className="main">{children}</section>
    </div>
  );
}
