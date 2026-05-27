import { useEffect, useMemo, useRef, useState } from "react";
import {
  exportJob,
  getJob,
  health,
  subscribe,
  uploadJob,
  type ExportFormat,
  type JobProgress,
  type JobResult,
  type Segment,
} from "./api";

const FORMATS: { id: ExportFormat; label: string; mime: string; ext: string }[] = [
  { id: "srt", label: "SRT", mime: "text/plain", ext: "srt" },
  { id: "vtt", label: "VTT", mime: "text/vtt", ext: "vtt" },
  { id: "txt", label: "TXT", mime: "text/plain", ext: "txt" },
  { id: "md", label: "Markdown", mime: "text/markdown", ext: "md" },
  { id: "json", label: "JSON", mime: "application/json", ext: "json" },
];

function fmt(t: number) {
  const m = Math.floor(t / 60).toString().padStart(2, "0");
  const s = Math.floor(t % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

function speakerClass(spk: string | null | undefined) {
  if (!spk) return "";
  const n = parseInt(spk.replace(/\D/g, ""), 10);
  return `spk spk-${isNaN(n) ? 0 : n % 5}`;
}

export function App() {
  const [file, setFile] = useState<File | null>(null);
  const [diarize, setDiarize] = useState(true);
  const [language, setLanguage] = useState("");
  const [progress, setProgress] = useState<JobProgress | null>(null);
  const [job, setJob] = useState<JobResult | null>(null);
  const [info, setInfo] = useState<{ diarize_available: boolean; model: string; device: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const unsubRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    health().then(setInfo).catch(() => setInfo(null));
    return () => unsubRef.current?.();
  }, []);

  async function onStart() {
    if (!file) return;
    setBusy(true);
    setJob(null);
    setProgress(null);
    try {
      const { id } = await uploadJob({
        file,
        diarize: diarize && !!info?.diarize_available,
        language: language || undefined,
      });
      unsubRef.current?.();
      unsubRef.current = subscribe(id, async (p) => {
        setProgress(p);
        if (p.status === "done" || p.status === "error") {
          const j = await getJob(id);
          setJob(j);
          setBusy(false);
        }
      });
    } catch (e: any) {
      setProgress({
        id: "",
        status: "error",
        progress: 0,
        message: "",
        error: e?.message || String(e),
      });
      setBusy(false);
    }
  }

  async function downloadAs(format: ExportFormat) {
    if (!job) return;
    const f = FORMATS.find((x) => x.id === format)!;
    const content = await exportJob(job.id, format);
    const blob = new Blob([content], { type: f.mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${job.filename}.${f.ext}`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function copyText() {
    if (!job?.result) return;
    const text = job.result.segments
      .map((s) => (s.speaker ? `[${s.speaker}] ${s.text}` : s.text))
      .join("\n");
    navigator.clipboard.writeText(text);
  }

  const segments: Segment[] = job?.result?.segments ?? [];
  const pct = useMemo(() => Math.round((progress?.progress ?? 0) * 100), [progress]);

  return (
    <div className="app">
      <h1>Transcript</h1>

      <div className="card">
        <div className="row">
          <input
            type="file"
            accept="audio/*,video/*"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
          <label>
            <input
              type="checkbox"
              checked={diarize}
              disabled={!info?.diarize_available}
              onChange={(e) => setDiarize(e.target.checked)}
            />{" "}
            Спикеры {info && !info.diarize_available && "(нет HF_TOKEN)"}
          </label>
          <label>
            Язык:{" "}
            <input
              type="text"
              placeholder="auto"
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              style={{ width: 70 }}
            />
          </label>
          <button disabled={!file || busy} onClick={onStart}>
            {busy ? "..." : "Транскрибировать"}
          </button>
        </div>
        {info && (
          <div className="status" style={{ marginTop: 10 }}>
            model: {info.model} · device: {info.device} ·{" "}
            diarize: {info.diarize_available ? "on" : "off"}
          </div>
        )}
      </div>

      {progress && (
        <div className="card">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <strong>{progress.status}</strong>
            <span>{pct}%</span>
          </div>
          <div className="progress"><div className="bar" style={{ width: `${pct}%` }} /></div>
          <div className={`status ${progress.error ? "error" : ""}`}>
            {progress.error || progress.message}
          </div>
        </div>
      )}

      {job?.result && (
        <div className="card">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <div>
              <strong>{job.filename}</strong>{" "}
              <span className="status">
                · {job.result.language} · {fmt(job.result.duration)} · {segments.length} сегментов
              </span>
            </div>
          </div>
          <div className="toolbar">
            <button onClick={copyText}>Copy</button>
            {FORMATS.map((f) => (
              <button key={f.id} onClick={() => downloadAs(f.id)}>{f.label}</button>
            ))}
          </div>
          <div className="segments" style={{ marginTop: 12 }}>
            {segments.map((s, i) => (
              <div className="seg" key={i}>
                <div className="ts">{fmt(s.start)}</div>
                <div className={speakerClass(s.speaker)}>{s.speaker ?? ""}</div>
                <div>{s.text}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
