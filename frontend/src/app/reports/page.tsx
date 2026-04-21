"use client";
import { useState, useEffect } from "react";
import Link from "next/link";

type Report = { filename: string; url: string; size_kb: number; date: string; };
function ReportRow({ report, onDeleted }: { report: Report; onDeleted: () => void }) {
  const [deleting, setDeleting] = useState(false);
  const [showModal, setShowModal] = useState(false);

  const handleDelete = async () => {
    setDeleting(true);
    try {
      const res = await fetch(`http://localhost:8000/delete-report/${encodeURIComponent(report.filename)}`, {
        method: "DELETE",
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `Server error: ${res.status}`);
      setShowModal(false);
      onDeleted();
    } catch (err: any) {
      alert(`Failed to delete: ${err.message}`);
    } finally {
      setDeleting(false);
    }
  };

  return (
    <>
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "1rem 1.25rem", borderRadius: "12px",
        background: "var(--surface-container-high)",
        border: "1px solid rgba(255,255,255,.05)",
        gap: "1rem", flexWrap: "wrap",
        opacity: deleting ? 0.4 : 1, transition: "opacity .2s",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          <div style={{ fontSize: "2rem" }}>📄</div>
          <div>
            <div style={{ fontWeight: 600, fontSize: "0.9rem" }}>{report.filename}</div>
            <div className="label-sm">{report.date} · {report.size_kb} KB</div>
          </div>
        </div>
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
          <a href={`http://localhost:8000${report.url}`} target="_blank" rel="noopener noreferrer">
            <button className="btn btn-secondary" style={{ fontSize: "0.8rem", padding: "0.45rem 1rem" }}>
              📖 View
            </button>
          </a>
          <a href={`http://localhost:8000${report.url}`} download={report.filename}>
            <button className="btn btn-ghost" style={{ fontSize: "0.8rem", padding: "0.45rem 1rem" }}>
              ⬇ Download
            </button>
          </a>
          <button
            onClick={() => setShowModal(true)}
            disabled={deleting}
            style={{
              fontSize: "0.8rem", padding: "0.45rem 0.85rem", border: "none", borderRadius: "8px",
              cursor: "pointer", fontWeight: 600,
              background: "rgba(239,68,68,.12)", color: "#ef4444",
              transition: "background .2s",
            }}
            onMouseEnter={e => (e.currentTarget.style.background = "rgba(239,68,68,.25)")}
            onMouseLeave={e => (e.currentTarget.style.background = "rgba(239,68,68,.12)")}
          >
            {deleting ? "…" : "🗑 Delete"}
          </button>
        </div>
      </div>

      {}
      {showModal && (
        <div style={{
          position: "fixed", top: 0, left: 0, right: 0, bottom: 0,
          background: "rgba(0,0,0,0.6)", backdropFilter: "blur(4px)",
          display: "flex", alignItems: "center", justifyContent: "center", zIndex: 9999
        }}>
          <div className="glass-card" style={{ width: "100%", maxWidth: "420px", padding: "1.5rem" }}>
            <div className="title-md" style={{ marginBottom: "0.5rem" }}>Delete Report?</div>
            <div style={{ fontSize: "0.9rem", color: "var(--on-surface-variant)", marginBottom: "1.25rem", lineHeight: 1.5 }}>
              Are you sure you want to permanently delete <strong>{report.filename}</strong>? This action cannot be undone.
            </div>
            <div style={{ display: "flex", gap: "0.75rem", justifyContent: "flex-end" }}>
              <button className="btn btn-ghost" onClick={() => setShowModal(false)} disabled={deleting}>
                Cancel
              </button>
              <button 
                className="btn btn-amber" 
                style={{ background: "#ef4444", color: "white" }} 
                onClick={handleDelete} 
                disabled={deleting}
              >
                {deleting ? "Deleting..." : "Yes, Delete"}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
export default function ReportsPage() {
  const [reports, setReports] = useState<Report[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchReports = () => {
    setLoading(true);
    fetch("http://localhost:8000/stats")
      .then(r => r.json())
      .then(d => setReports(d.reports ?? []))
      .catch(() => setReports([]))
      .finally(() => setLoading(false));
  };

  useEffect(() => { fetchReports(); }, []);

  return (
    <>
      <div>
        <div className="label-sm">Compliance Documentation</div>
        <h1 className="headline" style={{ marginTop: "0.25rem" }}>Reports</h1>
      </div>

      {}
      <div className="glass-card">
        <div className="title-md" style={{ marginBottom: "1rem" }}>📄 Generate a Compliance Report</div>
        <div style={{
          padding: "1rem 1.25rem", borderRadius: "10px",
          background: "rgba(59,130,246,.08)", border: "1px solid rgba(59,130,246,.2)",
          fontSize: "0.85rem", color: "var(--on-surface-variant)", lineHeight: 1.7,
        }}>
          <strong style={{ color: "#60a5fa" }}>How to generate a report:</strong><br />
          1. Go to <a href="/analysis" style={{ color: "var(--primary)", fontWeight: 600 }}>Live Analysis</a> and upload T1 / T2 TIFF images.<br />
          2. Click <strong>Run AI Inference</strong> to analyse the selected zone.<br />
          3. Click <strong>📄 Download PDF Report</strong> — the file appears in the list below instantly.
        </div>
        <div style={{ marginTop: "1rem", display: "flex", gap: "0.75rem" }}>
          <a href="/analysis" style={{ textDecoration: "none" }}>
            <button className="btn btn-primary">🔬 Go to Live Analysis →</button>
          </a>
          <button className="btn btn-ghost" onClick={fetchReports} style={{ fontSize: "0.85rem" }}>
            ↻ Refresh List
          </button>
        </div>
      </div>

      {}
      <div className="glass-card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.25rem" }}>
          <div className="title-md">Saved Reports ({reports.length})</div>
          <button className="btn btn-ghost" style={{ fontSize: "0.8rem", padding: "0.4rem 0.9rem" }} onClick={fetchReports}>
            ↻ Refresh
          </button>
        </div>

        {loading && (
          <div style={{ textAlign: "center", color: "var(--on-surface-variant)", padding: "2rem" }}>
            Loading reports…
          </div>
        )}

        {!loading && reports.length === 0 && (
          <div style={{ textAlign: "center", padding: "3rem", color: "var(--on-surface-variant)" }}>
            <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>📭</div>
            <div className="title-md" style={{ marginBottom: "0.5rem" }}>No reports yet</div>
            <div style={{ fontSize: "0.85rem", marginBottom: "1.5rem" }}>
              Generate a report from the Live Analysis page after running inference.
            </div>
            <Link href="/analysis">
              <button className="btn btn-primary">Go to Live Analysis →</button>
            </Link>
          </div>
        )}

        {!loading && reports.length > 0 && (
          <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
            {reports.map((r, i) => (
              <ReportRow key={r.filename + i} report={r} onDeleted={fetchReports} />
            ))}
          </div>
        )}
      </div>

      {}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "1rem" }}>
        {[
          { label: "Total Reports",  value: reports.length, badge: "LIVE", color: "var(--primary)" },
          { label: "Zones Monitored", value: 3,             badge: "CONFIGURED", color: "#3b82f6" },
          { label: "Storage Used",   value: `${reports.reduce((s, r) => s + r.size_kb, 0).toFixed(1)} KB`, badge: "LIVE", color: "var(--primary)" },
        ].map(c => (
          <div key={c.label} className="glass-card" style={{ textAlign: "center" }}>
            <div className="label-sm">{c.label}</div>
            <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontWeight: 700, fontSize: "2rem", margin: "0.5rem 0", color: c.color }}>
              {c.value}
            </div>
            <div style={{ fontSize: "0.65rem", fontWeight: 700, letterSpacing: "0.08em", padding: "2px 8px", borderRadius: 99, display: "inline-block", background: "rgba(255,255,255,.06)", color: "var(--on-surface-variant)" }}>
              {c.badge}
            </div>
          </div>
        ))}
      </div>
    </>
  );
}
