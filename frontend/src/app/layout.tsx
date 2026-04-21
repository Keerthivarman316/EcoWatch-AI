"use client";
import { useState, ReactNode } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import "./globals.css";

const navItems = [
  { href: "/",         label: "Dashboard",     icon: "📊" },
  { href: "/analysis", label: "Live Analysis", icon: "🔬" },
  { href: "/reports",  label: "Reports",       icon: "📄" },
];

export default function RootLayout({ children }: { children: ReactNode }) {
  const pathname  = usePathname();
  const [open, setOpen] = useState(false);

  return (
    <html lang="en">
      <head>
        <meta charSet="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>EcoWatch AI | Environmental Compliance Platform</title>
        <meta name="description" content="AI-powered satellite change detection for industrial environmental compliance monitoring." />
      </head>
      <body>
        {}
        <button className="sidebar-toggle" onClick={() => setOpen(o => !o)} aria-label="Toggle menu">☰</button>

        <div className="app-shell">
          {}
          <aside className={`sidebar ${open ? "open" : ""}`} onClick={() => setOpen(false)}>
            <div className="sidebar-logo">
              <span style={{ fontSize: "1.5rem" }}>🌿</span>
              <div>
                EcoWatch AI
                <span>Synthetic Observatory</span>
              </div>
            </div>
            {navItems.map(item => (
              <Link
                key={item.href}
                href={item.href}
                className={`nav-item ${pathname === item.href ? "active" : ""}`}
              >
                <span className="nav-icon">{item.icon}</span>
                {item.label}
              </Link>
            ))}
            <div style={{ marginTop: "auto" }}>
              <button className="nav-item" style={{ color: "var(--outline)" }}>
                <span className="nav-icon">⚙️</span> Settings
              </button>
            </div>
          </aside>

          {}
          <main className="main">{children}</main>
        </div>
      </body>
    </html>
  );
}
