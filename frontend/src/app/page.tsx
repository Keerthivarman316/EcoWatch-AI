"use client";
import { useState, useEffect } from "react";
import Link from "next/link";

function StatCard({ label, value, badgeText, badgeClass, sub, isLive = false }: {
  label: string; value: string | number;
  badgeText?: string; badgeClass?: string; sub?: string; isLive?: boolean;
}) {
  return (
    <div className="glass-card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div className="label-sm">{label}</div>
        {isLive
          ? <span style={{ fontSize: "0.6rem", fontWeight: 700, letterSpacing: "0.08em", padding: "2px 7px", borderRadius: 99, background: "rgba(78,222,163,.15)", color: "var(--primary)", textTransform: "uppercase" }}>● LIVE</span>
          : <span style={{ fontSize: "0.6rem", fontWeight: 600, letterSpacing: "0.08em", padding: "2px 7px", borderRadius: 99, background: "rgba(255,255,255,.06)", color: "var(--on-surface-variant)", textTransform: "uppercase" }}>DEMO</span>
        }
      </div>
      <div className="stat-value" style={{ marginTop: "0.5rem" }}>{value}</div>
      {sub && <div className="body-md" style={{ marginTop: "0.25rem", fontSize: "0.8rem" }}>{sub}</div>}
      {badgeText && <div className={`stat-badge ${badgeClass}`}>{badgeText}</div>}
    </div>
  );
}

function ZoneMap() {
  return (
    <div className="glass-card" style={{ height: "100%" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
        <div className="title-md">Zone Status Map</div>
        <span style={{ fontSize: "0.65rem", color: "var(--on-surface-variant)", background: "rgba(255,255,255,.06)", padding: "2px 8px", borderRadius: 99 }}>
          DEMO — approximate zone positions
        </span>
      </div>
      <div className="map-placeholder" style={{ minHeight: "260px" }}>
        <div className="map-grid-lines" />
        <div className="zone-marker" style={{ left: "30%", top: "40%" }}>
          <div className="zone-dot" style={{ background: "#4edea3", color: "#4edea3" }} />
          <div className="zone-label" style={{ color: "#4edea3" }}>Peenya</div>
        </div>
        <div className="zone-marker" style={{ left: "55%", top: "60%" }}>
          <div className="zone-dot" style={{ background: "#3b82f6", color: "#3b82f6" }} />
          <div className="zone-label" style={{ color: "#3b82f6" }}>Bommasandra</div>
        </div>
        <div className="zone-marker" style={{ left: "75%", top: "75%" }}>
          <div className="zone-dot" style={{ background: "#f59e0b", color: "#f59e0b" }} />
          <div className="zone-label" style={{ color: "#f59e0b" }}>Nanjangud</div>
        </div>
        <div style={{ position: "absolute", left: "25%", top: "35%", width: 80, height: 60, borderRadius: "50%", background: "rgba(78,222,163,.15)", filter: "blur(20px)" }} />
        <div style={{ position: "absolute", left: "50%", top: "55%", width: 100, height: 70, borderRadius: "50%", background: "rgba(59,130,246,.15)", filter: "blur(24px)" }} />
        <div style={{ position: "absolute", left: "70%", top: "70%", width: 70, height: 50, borderRadius: "50%", background: "rgba(245,158,11,.15)", filter: "blur(18px)" }} />
        <div style={{ position: "absolute", bottom: "0.75rem", right: "0.75rem", background: "rgba(16,20,26,.8)", borderRadius: "8px", padding: "0.5rem 0.75rem" }}>
          <div className="label-sm" style={{ marginBottom: "0.4rem" }}>Severity</div>
          {[["Low", "#4edea3"], ["Medium", "#3b82f6"], ["High", "#f59e0b"], ["Critical", "#ef4444"]].map(([l, c]) => (
            <div key={l} style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "2px" }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: c }} />
              <span style={{ fontSize: "0.65rem", color: "var(--on-surface-variant)" }}>{l}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

type Stats = { complaint_count: number; report_count: number; reports: unknown[] };

export default function Dashboard() {
  const [stats, setStats] = useState<Stats>({ complaint_count: 0, report_count: 0, reports: [] });
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);

  useEffect(() => {
    fetch("http://localhost:8000/stats")
      .then(r => r.json())
      .then((d: Stats) => { setStats(d); setApiOnline(true); })
      .catch(() => setApiOnline(false));
  }, []);

  return (
    <>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", flexWrap: "wrap", gap: "0.5rem" }}>
        <div>
          <div className="label-sm">Environmental Compliance Hub</div>
          <h1 className="headline" style={{ marginTop: "0.25rem" }}>Synthetic Observatory</h1>
        </div>
        {apiOnline !== null && (
          <div style={{ fontSize: "0.75rem", padding: "4px 12px", borderRadius: 99, fontWeight: 600,
            background: apiOnline ? "rgba(78,222,163,.12)" : "rgba(239,68,68,.12)",
            color: apiOnline ? "var(--primary)" : "#ef4444" }}>
            {apiOnline ? "● API Online" : "● API Offline"}
          </div>
        )}
      </div>

      <div className="stat-grid">
        <StatCard label="Violations Detected" value="—" badgeText="Run inference to detect" badgeClass="badge-blue" isLive={false} />
        <StatCard label="Active Zones" value="3" sub="Peenya · Bommasandra · Nanjangud" badgeText="● Configured" badgeClass="badge-green" isLive={false} />
        <StatCard label="Complaints Filed" value={stats.complaint_count}
          badgeText={stats.complaint_count === 0 ? "No complaints yet" : `${stats.complaint_count} on record`}
          badgeClass="badge-amber" isLive={true} />
        <StatCard label="Reports Generated" value={stats.report_count}
          badgeText={stats.report_count === 0 ? "No PDFs yet" : `${stats.report_count} report(s) on disk`}
          badgeClass="badge-blue" isLive={true} />
      </div>

      {}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
        <Link href="/analysis" style={{ textDecoration: "none" }}>
          <div className="glass-card" style={{ cursor: "pointer", border: "1px solid rgba(78,222,163,.15)", transition: "border-color .2s" }}
            onMouseEnter={e => (e.currentTarget.style.borderColor = "rgba(78,222,163,.45)")}
            onMouseLeave={e => (e.currentTarget.style.borderColor = "rgba(78,222,163,.15)")}>
            <div style={{ fontSize: "2rem", marginBottom: "0.75rem" }}>🛰️</div>
            <div className="title-md" style={{ marginBottom: "0.4rem" }}>Run Live Analysis</div>
            <div style={{ fontSize: "0.82rem", color: "var(--on-surface-variant)" }}>
              Upload T1 / T2 satellite images, run AI inference and detect green belt violations in real‑time.
            </div>
          </div>
        </Link>
        <Link href="/reports" style={{ textDecoration: "none" }}>
          <div className="glass-card" style={{ cursor: "pointer", border: "1px solid rgba(59,130,246,.15)", transition: "border-color .2s" }}
            onMouseEnter={e => (e.currentTarget.style.borderColor = "rgba(59,130,246,.45)")}
            onMouseLeave={e => (e.currentTarget.style.borderColor = "rgba(59,130,246,.15)")}>
            <div style={{ fontSize: "2rem", marginBottom: "0.75rem" }}>📄</div>
            <div className="title-md" style={{ marginBottom: "0.4rem" }}>View Reports</div>
            <div style={{ fontSize: "0.82rem", color: "var(--on-surface-variant)" }}>
              Browse, view, and download all generated PDF compliance reports saved to disk.
            </div>
          </div>
        </Link>
      </div>

      {}
      <ZoneMap />
    </>
  );
}
