export const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

export type Segment = {
  start: number;
  end: number;
  text: string;
  speaker?: string | null;
};

export type JobStatus =
  | "queued"
  | "extracting"
  | "transcribing"
  | "aligning"
  | "diarizing"
  | "done"
  | "error";

export type JobProgress = {
  id: string;
  status: JobStatus;
  progress: number;
  message: string;
  error: string | null;
};

export type JobResult = {
  id: string;
  filename: string;
  status: JobStatus;
  progress: number;
  message: string;
  error: string | null;
  language: string | null;
  diarize: boolean;
  result: {
    language: string;
    duration: number;
    segments: Segment[];
  } | null;
};

export async function health() {
  const r = await fetch(`${API}/api/health`);
  return r.json() as Promise<{
    ok: boolean;
    model: string;
    device: string;
    diarize_available: boolean;
  }>;
}

export async function uploadJob(opts: {
  file: File;
  diarize: boolean;
  language?: string;
}): Promise<{ id: string; diarize: boolean }> {
  const fd = new FormData();
  fd.append("file", opts.file);
  fd.append("diarize", String(opts.diarize));
  if (opts.language) fd.append("language", opts.language);
  const r = await fetch(`${API}/api/jobs`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`upload failed: ${r.status}`);
  return r.json();
}

export function subscribe(jobId: string, onProgress: (p: JobProgress) => void) {
  const es = new EventSource(`${API}/api/jobs/${jobId}/events`);
  es.addEventListener("progress", (e) => {
    onProgress(JSON.parse((e as MessageEvent).data));
  });
  es.onerror = () => es.close();
  return () => es.close();
}

export async function getJob(jobId: string): Promise<JobResult> {
  const r = await fetch(`${API}/api/jobs/${jobId}`);
  return r.json();
}

export async function getSrt(jobId: string): Promise<string> {
  const r = await fetch(`${API}/api/jobs/${jobId}/srt`);
  const j = await r.json();
  return j.srt as string;
}

export type ExportFormat = "srt" | "vtt" | "txt" | "md" | "json";

export async function exportJob(jobId: string, format: ExportFormat): Promise<string> {
  const r = await fetch(`${API}/api/jobs/${jobId}/export?format=${format}`);
  const j = await r.json();
  return j.content as string;
}
