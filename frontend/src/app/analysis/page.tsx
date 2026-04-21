"use client";
import { useState, useEffect } from "react";

type Blob = {
  id: string; severity: "High" | "Medium" | "Low";
  area_km2: number; lat: number; lon: number;
};
type PredResult = {
  violations: number;
  confidence: number;
  coverage_km2: number;
  heatmap_url: string;
  comparison_url?: string;
  blobs: Blob[];
  analysis_type: string;
  zone: string;
};

const SEV_CHIP: Record<string, string> = { High: "CRITICAL", Medium: "REVIEW", Low: "LOW" };
const SEV_CLR:  Record<string, string> = {
  High: "#ef4444", Medium: "#f59e0b", Low: "#4edea3",
};

export default function LiveAnalysis() {
  const [t1, setT1]       = useState<File | null>(null);
  const [t2, setT2]       = useState<File | null>(null);
  const [zone, setZone]   = useState("Bommasandra");
  const [analysis, setAnalysis] = useState("Change Detection");
  const [loading, setLoading]   = useState(false);
  const [result, setResult]     = useState<PredResult | null>(null);
  const [error, setError]       = useState<string | null>(null);
  const [apiStatus, setApiStatus] = useState<"ok"|"err"|null>(null);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [pdfError, setPdfError]     = useState<string | null>(null);
  const [imgTab, setImgTab]         = useState<"compare"|"heatmap">("compare");
  useEffect(() => {
    try {
      const saved = sessionStorage.getItem("eco_analysis_state");
      if (saved) {
        const parsed = JSON.parse(saved);
        if (parsed.result) setResult(parsed.result);
        if (parsed.zone) setZone(parsed.zone);
        if (parsed.analysis) setAnalysis(parsed.analysis);
      }
    } catch {}
  }, []);

  // Save to sessionStorage when results/config update
  useEffect(() => {
    if (result) {
      sessionStorage.setItem("eco_analysis_state", JSON.stringify({ result, zone, analysis }));
    }
  }, [result, zone, analysis]);

  const checkApi = async (): Promise<boolean> => {
    try {
      const r = await fetch("http://localhost:8000/health", { signal: AbortSignal.timeout(3000) });
      if (r.ok) { setApiStatus("ok"); return true; }
    } catch {}
    setApiStatus("err"); return false;
  };

  const runInference = async () => {
    if (!t1 || !t2) { setError("Please upload both T1 and T2 TIFF files."); return; }
    setLoading(true); setError(null); setResult(null); setPdfError(null);
    const alive = await checkApi();
    if (!alive) {
      setError("Cannot reach backend (http://localhost:8000). Start it:\n  cd e:\\EcoWatch-AI\\src && python api.py");
      setLoading(false); return;
    }
    const fd = new FormData();
    fd.append("t1_file", t1); fd.append("t2_file", t2);
    fd.append("analysis_type", analysis); fd.append("zone_name", zone);
    try {
      const res  = await fetch("http://localhost:8000/predict", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? `Server error ${res.status}`);
      setResult(data);
      setImgTab("compare"); // switch to comparison view on new result
    } catch (e: any) {
      const msg = e.message ?? String(e);
      setError(msg.includes("fetch") ? "API disconnected. Restart:\n  python e:\\EcoWatch-AI\\src\\api.py" : msg);
    } finally {
      setLoading(false);
    }
  };

  const downloadReport = async () => {
    if (!result) {
      setPdfError("Run AI Inference first, then click Download PDF Report.");
      return;
    }
    setPdfLoading(true); setPdfError(null);
    try {
      const fd = new FormData();
      fd.append("zone_name",     result.zone);
      fd.append("analysis_type", result.analysis_type);
      fd.append("violations",    String(result.violations));
      fd.append("confidence",    String(result.confidence));
      fd.append("coverage_km2",  String(result.coverage_km2));
      fd.append("blobs_json",    JSON.stringify(result.blobs ?? []));
      fd.append("heatmap_url",   result.heatmap_url ?? "");
      const res  = await fetch("http://localhost:8000/generate-report", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? `Error ${res.status}`);
      const pdfRes = await fetch(`http://localhost:8000${data.report_url}`);
      const pdfBlob = await pdfRes.blob();
      const url = URL.createObjectURL(pdfBlob);
      const a = document.createElement("a");
      a.href = url;
      a.download = data.report_url.split('/').pop() || "Compliance_Report.pdf";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

    } catch (e: any) {
      setPdfError(e.message ?? "Report generation failed");
    } finally {
      setPdfLoading(false);
    }
  };

  const ts = Date.now(); // cache-buster

  return (
    <>
      {}
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-end", flexWrap:"wrap", gap:"0.5rem" }}>
        <div>
          <div className="label-sm">Satellite Analysis</div>
          <h1 className="headline" style={{ marginTop:"0.25rem" }}>Live Analysis</h1>
        </div>
        {apiStatus !== null && (
          <div style={{ fontSize:"0.75rem", padding:"4px 12px", borderRadius:99, fontWeight:600,
            background: apiStatus==="ok" ? "rgba(78,222,163,.12)" : "rgba(239,68,68,.12)",
            color: apiStatus==="ok" ? "var(--primary)" : "#ef4444" }}>
            {apiStatus==="ok" ? "● API Online" : "● API Offline"}
          </div>
        )}
      </div>

      {}
      <div className="glass-card">
        <div className="title-md" style={{ marginBottom:"1.25rem" }}>Upload Satellite Images</div>
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"1.25rem" }}>
          <UploadZone label="T1 — Prior Image (TIFF)"   file={t1}
            onDrop={e=>{ e.preventDefault(); setT1(e.dataTransfer.files[0]??null); }}
            onChange={e=>setT1(e.target.files?.[0]??null)} inputId="t1-input" />
          <UploadZone label="T2 — Current Image (TIFF)" file={t2}
            onDrop={e=>{ e.preventDefault(); setT2(e.dataTransfer.files[0]??null); }}
            onChange={e=>setT2(e.target.files?.[0]??null)} inputId="t2-input" />
        </div>

        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr auto", gap:"1rem", marginTop:"1.5rem", alignItems:"flex-end" }}>
          <div className="field" style={{ marginBottom:0 }}>
            <label>Zone Name</label>
            <select value={zone} onChange={e=>setZone(e.target.value)}>
              {["Peenya","Bommasandra","Nanjangud"].map(z=><option key={z}>{z}</option>)}
            </select>
          </div>
          <div className="field" style={{ marginBottom:0 }}>
            <label>Analysis Type</label>
            <select value={analysis} onChange={e=>{ setAnalysis(e.target.value); setResult(null); setPdfError(null); }}>
              <option>Change Detection</option>
              <option>Vegetation Segmentation</option>
            </select>
          </div>
          <button className="btn btn-primary btn-lg" onClick={runInference} disabled={loading} style={{ whiteSpace:"nowrap" }}>
            {loading ? "⏳ Analyzing…" : "🧠 Run AI Inference"}
          </button>
        </div>

        <div style={{ marginTop:"0.75rem", padding:"0.6rem 1rem", borderRadius:"8px",
          background:"rgba(59,130,246,.08)", fontSize:"0.8rem", color:"var(--on-surface-variant)" }}>
          {analysis==="Change Detection"
            ? "ℹ️ Change Detection: computes pixel-wise spectral difference (T2−T1) across all bands."
            : "ℹ️ Vegetation Segmentation: computes NDVI loss (NIR−Red)/(NIR+Red) between T1 and T2 — no trained model required."}
        </div>

        {error && (
          <div style={{ marginTop:"1rem", padding:"0.85rem 1rem", background:"rgba(239,68,68,.1)",
            border:"1px solid rgba(239,68,68,.3)", borderRadius:"8px", color:"#ef4444",
            fontSize:"0.85rem", whiteSpace:"pre-line" }}>
            ⚠️ {error}
          </div>
        )}
      </div>

      {}
      {loading && (
        <div className="glass-card" style={{ textAlign:"center", padding:"3rem" }}>
          <div style={{ fontSize:"3rem", marginBottom:"1rem", animation:"pulse 1.5s infinite" }}>🛰️</div>
          <div className="title-md">
            {analysis==="Vegetation Segmentation" ? "Computing NDVI Loss Map…" : "Running Spectral Change Analysis…"}
          </div>
          <div className="body-md" style={{ marginTop:"0.5rem" }}>Analysing {zone} · {analysis}</div>
        </div>
      )}

      {}
      {result && !loading && (
        <div style={{ display:"grid", gridTemplateColumns:"3fr 2fr", gap:"1.25rem" }}>

          {}
          <div className="glass-card">
            {}
            <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:"1rem" }}>
              <div style={{ display:"flex", gap:"0.5rem" }}>
                {(["compare","heatmap"] as const).map(tab=>(
                  <button key={tab} onClick={()=>setImgTab(tab)}
                    style={{ padding:"4px 14px", borderRadius:99, border:"none", cursor:"pointer",
                      fontWeight:600, fontSize:"0.75rem",
                      background: imgTab===tab ? "var(--primary)" : "rgba(255,255,255,.07)",
                      color: imgTab===tab ? "#0d1117" : "var(--on-surface-variant)" }}>
                    {tab==="compare" ? "🔍 Comparison" : "🌡 Heatmap"}
                  </button>
                ))}
              </div>
              <span style={{ fontSize:"0.65rem", color:"var(--primary)", background:"rgba(78,222,163,.1)",
                padding:"2px 8px", borderRadius:99, fontWeight:600 }}>LIVE</span>
            </div>

            {imgTab==="compare" && result.comparison_url ? (
              <>
                <img
                  src={`http://localhost:8000${result.comparison_url}?t=${ts}`}
                  alt="Before / Change Map / After Comparison"
                  style={{ width:"100%", borderRadius:"0.75rem", display:"block" }}
                />
                <div className="label-sm" style={{ marginTop:"0.6rem" }}>
                  🟢 Left = T1 Before · 🌡 Centre = AI Change Map · 🔴 Right = T2 After (neon outlines = violations)
                </div>
              </>
            ) : (
              <>
                <img
                  src={`http://localhost:8000${result.heatmap_url}?t=${ts}`}
                  alt="Analysis Heatmap"
                  style={{ width:"100%", borderRadius:"0.75rem", display:"block" }}
                />
                <div className="label-sm" style={{ marginTop:"0.6rem" }}>
                  {result.analysis_type==="Vegetation Segmentation"
                    ? "Bright = vegetation loss detected"
                    : "Blue = no change | Red/Yellow = change detected"}
                </div>
              </>
            )}
          </div>

          {}
          <div className="glass-card">
            <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:"1rem" }}>
              <div className="title-md">Detection Summary</div>
              <span style={{ fontSize:"0.65rem", color:"var(--primary)", background:"rgba(78,222,163,.1)",
                padding:"2px 8px", borderRadius:99, fontWeight:600 }}>LIVE</span>
            </div>

            <div style={{ textAlign:"center", padding:"0.75rem 0 1.25rem" }}>
              <div className="label-sm">Violations Detected</div>
              <div style={{ fontFamily:"'Space Grotesk',sans-serif", fontWeight:700, fontSize:"3rem",
                color: result.violations>0 ? "#ef4444" : "#4edea3", lineHeight:1 }}>
                {result.violations}
              </div>
            </div>

            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"0.6rem", marginBottom:"1.25rem" }}>
              {[
                { label:"Confidence",    value:`${result.confidence.toFixed(1)}%` },
                { label:"Coverage Area", value:`${result.coverage_km2} km²` },
                { label:"Zone",          value:result.zone },
                { label:"Analysis",      value:result.analysis_type },
              ].map(m=>(
                <div key={m.label} style={{ padding:"0.65rem 0.75rem",
                  background:"var(--surface-container-low)", borderRadius:"8px" }}>
                  <div className="label-sm">{m.label}</div>
                  <div style={{ fontWeight:600, fontSize:"0.9rem", marginTop:"0.2rem" }}>{m.value}</div>
                </div>
              ))}
            </div>

            {result.blobs.length>0 && (
              <>
                <div className="label-sm" style={{ marginBottom:"0.5rem" }}>Detected Regions (real coords)</div>
                <div style={{ maxHeight:"190px", overflowY:"auto", display:"flex", flexDirection:"column", gap:4 }}>
                  {result.blobs.map(b=>(
                    <div key={b.id} style={{ display:"flex", justifyContent:"space-between", alignItems:"center",
                      padding:"0.5rem 0.75rem", background:"var(--surface-container-high)", borderRadius:"8px" }}>
                      <div>
                        <div style={{ fontWeight:600, fontSize:"0.82rem" }}>{b.id}</div>
                        <div style={{ fontSize:"0.7rem", color:"var(--on-surface-variant)", marginTop:1 }}>
                          {b.lat}°N, {b.lon}°E · {b.area_km2} km²
                        </div>
                      </div>
                      <span style={{ fontSize:"0.65rem", padding:"2px 8px", borderRadius:99,
                        fontWeight:700, background:`${SEV_CLR[b.severity]}22`, color:SEV_CLR[b.severity] }}>
                        {SEV_CHIP[b.severity]}
                      </span>
                    </div>
                  ))}
                </div>
              </>
            )}

            {result.violations===0 && (
              <div style={{ textAlign:"center", padding:"1rem", color:"var(--primary)", fontWeight:600, fontSize:"0.9rem" }}>
                ✅ No violations detected in this image pair.
              </div>
            )}

            {}
            <div style={{ marginTop:"1.25rem", display:"flex", flexDirection:"column", gap:"0.6rem" }}>
              <button
                className="btn btn-amber btn-full"
                onClick={downloadReport}
                disabled={pdfLoading}
              >
                {pdfLoading ? "⏳ Generating PDF…" : "📄 Download PDF Report"}
              </button>
              {pdfError && (
                <div style={{ fontSize:"0.78rem", color:"#ef4444", padding:"0.4rem 0.75rem",
                  background:"rgba(239,68,68,.1)", borderRadius:6, lineHeight:1.4 }}>
                  ⚠️ {pdfError}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {}
      {!result && !loading && !error && (
        <div className="glass-card" style={{ textAlign:"center", padding:"2.5rem", color:"var(--on-surface-variant)" }}>
          <div style={{ fontSize:"2.5rem", marginBottom:"1rem" }}>🛰️</div>
          <div className="title-md" style={{ marginBottom:"0.5rem" }}>Ready for Analysis</div>
          <div style={{ fontSize:"0.85rem", maxWidth:"480px", margin:"0 auto", lineHeight:1.6 }}>
            Upload T1 (prior) and T2 (current) TIFF files, then click <strong>Run AI Inference</strong>.<br />
            Sample files: <code style={{ background:"rgba(255,255,255,.06)", padding:"2px 6px", borderRadius:4 }}>
              e:\EcoWatch-AI\Data\Sample_TIFFs\
            </code>
          </div>
        </div>
      )}
    </>
  );
}

function UploadZone({ label, file, onDrop, onChange, inputId }: {
  label:string; file:File|null;
  onDrop:(e:React.DragEvent)=>void;
  onChange:(e:React.ChangeEvent<HTMLInputElement>)=>void;
  inputId:string;
}) {
  return (
    <label htmlFor={inputId}
      className={`upload-zone ${file?"has-file":""}`}
      onDrop={onDrop} onDragOver={e=>e.preventDefault()}
      style={{ cursor:"pointer" }}>
      <div style={{ fontSize:"2rem", marginBottom:"0.75rem" }}>{file?"✅":"🛰️"}</div>
      <div className="title-md" style={{ marginBottom:"0.4rem" }}>{label}</div>
      {file
        ? <div style={{ color:"var(--primary)", fontWeight:600, fontSize:"0.875rem" }}>{file.name}</div>
        : <div style={{ fontSize:"0.8rem", color:"var(--on-surface-variant)" }}>Drop TIFF file or click to browse</div>}
      <input id={inputId} type="file" accept=".tif,.tiff" onChange={onChange} style={{ display:"none" }} />
    </label>
  );
}
