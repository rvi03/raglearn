"use client";

import { type ReactNode, createContext, useContext, useEffect, useState } from "react";

type Theme = "dark" | "light";
type ThemeContextValue = { theme: Theme; toggle: () => void };

const ThemeContext = createContext<ThemeContextValue | null>(null);
const STORAGE_KEY = "finrag-theme";

export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used within ThemeProvider");
  return ctx;
}

/**
 * Owns the `.cc dark|light` root class and the theme toggle. Defaults to dark
 * (matches SSR), then hydrates the saved choice from localStorage.
 */
export function ThemeProvider({ children }: { children: ReactNode }): ReactNode {
  const [theme, setTheme] = useState<Theme>("dark");

  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "light" || saved === "dark") setTheme(saved);
  }, []);

  const toggle = (): void => {
    setTheme((prev) => {
      const next: Theme = prev === "dark" ? "light" : "dark";
      localStorage.setItem(STORAGE_KEY, next);
      return next;
    });
  };

  return (
    <ThemeContext.Provider value={{ theme, toggle }}>
      <div className={`cc ${theme}`}>{children}</div>
    </ThemeContext.Provider>
  );
}
