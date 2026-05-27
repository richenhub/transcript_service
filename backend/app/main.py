import asyncio
import json
import os
import subprocess
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.serialization
_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat
torch.serialization.load = _torch_load_compat

for _mod_path in (
    "lightning_fabric.utilities.cloud_io",
    "pytorch_lightning.utilities.cloud_io",
    "pyannote.audio.core.io",
    "speechbrain.utils.checkpoints",
):
    try:
        import importlib
        _m = importlib.import_module(_mod_path)
        if hasattr(_m, "load"):
            _m.load = _torch_load_compat
        if hasattr(_m, "_load"):
            _m._load = _torch_load_compat
    except Exception:
        pass

try:
    import omegaconf
    import omegaconf.base
    import omegaconf.nodes
    import omegaconf.listconfig
    import omegaconf.dictconfig
    safe = [
        omegaconf.ListConfig,
        omegaconf.DictConfig,
        omegaconf.base.ContainerMetadata,
        omegaconf.base.Metadata,
        omegaconf.nodes.AnyNode,
        omegaconf.nodes.ValueNode,
        omegaconf.nodes.IntegerNode,
        omegaconf.nodes.FloatNode,
        omegaconf.nodes.StringNode,
        omegaconf.nodes.BooleanNode,
    ]
    torch.serialization.add_safe_globals(safe)
except Exception:
    pass

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

MODEL_NAME = os.environ.get("WHISPER_MODEL", "small")
DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
MODEL_CACHE = os.environ.get("MODEL_CACHE", "/models")
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or None
DIARIZE_DEFAULT = os.environ.get("DIARIZE", "true").lower() == "true"

DATA_DIR.mkdir(parents=True, exist_ok=True)

_models: dict = {}


def get_whisper_model():
    if "whisper" not in _models:
        import whisperx
        _models["whisper"] = whisperx.load_model(
            MODEL_NAME, DEVICE, compute_type=COMPUTE_TYPE, download_root=MODEL_CACHE
        )
    return _models["whisper"]


def get_align_model(language: str):
    import whisperx
    key = f"align_{language}"
    if key not in _models:
        model_a, metadata = whisperx.load_align_model(language_code=language, device=DEVICE)
        _models[key] = (model_a, metadata)
    return _models[key]


def get_diarize_pipeline():
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set. Diarization disabled.")
    if "diarize" not in _models:
        import whisperx
        _models["diarize"] = whisperx.DiarizationPipeline(
            use_auth_token=HF_TOKEN, device=DEVICE
        )
    return _models["diarize"]


@dataclass
class Job:
    id: str
    filename: str
    src_path: str = ""
    status: str = "queued"
    progress: float = 0.0
    message: str = ""
    result: Optional[dict] = None
    error: Optional[str] = None
    diarize: bool = False
    language: Optional[str] = None
    queue: asyncio.Queue = field(default_factory=asyncio.Queue, repr=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "src_path": self.src_path,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "diarize": self.diarize,
            "language": self.language,
            "result": self.result,
        }

    def persist(self):
        p = DATA_DIR / f"{self.id}.json"
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)

    def emit(self):
        payload = {
            "id": self.id,
            "status": self.status,
            "progress": round(self.progress, 3),
            "message": self.message,
            "error": self.error,
        }
        self.queue.put_nowait(payload)
        self.persist()


JOBS: dict[str, Job] = {}


def load_jobs_from_disk():
    for p in DATA_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            j = Job(
                id=d["id"],
                filename=d.get("filename", ""),
                src_path=d.get("src_path", ""),
                status=d.get("status", "error"),
                progress=d.get("progress", 0.0),
                message=d.get("message", ""),
                error=d.get("error"),
                diarize=d.get("diarize", False),
                language=d.get("language"),
                result=d.get("result"),
            )
            if j.status not in ("done", "error"):
                j.status = "error"
                j.error = "interrupted (backend restart)"
            JOBS[j.id] = j
        except Exception:
            pass


def extract_audio(src: Path, dst: Path):
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vn", "-ac", "1", "-ar", "16000",
        "-f", "wav", str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


async def _run_phase(job: Job, fn, *, label: str, p_start: float, p_end: float, eta: float):
    loop = asyncio.get_running_loop()
    stop = asyncio.event = asyncio.Event()

    async def ticker():
        t0 = loop.time()
        while not stop.is_set():
            elapsed = loop.time() - t0
            frac = min(0.99, elapsed / max(eta, 1.0))
            job.progress = p_start + (p_end - p_start) * frac
            job.message = f"{label} ~{int(elapsed)}s / ~{int(eta)}s"
            job.emit()
            try:
                await asyncio.wait_for(stop.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

    task = asyncio.create_task(ticker())
    try:
        return await loop.run_in_executor(None, fn)
    finally:
        stop.set()
        await task


async def run_job(job: Job, src_path: Path):
    loop = asyncio.get_running_loop()
    wav_path: Optional[Path] = None
    try:
        job.status = "extracting"
        job.message = "Extracting audio"
        job.progress = 0.02
        job.emit()

        wav_path = src_path.with_name(f"{src_path.stem}.16k.wav")
        await loop.run_in_executor(None, extract_audio, src_path, wav_path)

        import whisperx
        audio = await loop.run_in_executor(None, whisperx.load_audio, str(wav_path))
        duration = len(audio) / 16000.0

        rtf = {"tiny": 0.05, "base": 0.08, "small": 0.15, "medium": 0.35, "large-v3": 0.6}.get(MODEL_NAME, 0.2)
        eta_tr = max(5.0, duration * rtf)
        eta_al = max(3.0, duration * 0.05)
        eta_di = max(10.0, duration * 0.1)

        job.status = "transcribing"
        job.progress = 0.1
        job.emit()

        def _transcribe():
            model = get_whisper_model()
            return model.transcribe(audio, batch_size=8, language=job.language)

        result = await _run_phase(job, _transcribe,
                                  label="Transcribing", p_start=0.1, p_end=0.55, eta=eta_tr)

        detected_lang = result.get("language") or job.language or "en"
        job.language = detected_lang
        job.progress = 0.55
        job.message = f"Transcribed ({len(result.get('segments', []))} segments, lang={detected_lang})"
        job.emit()

        job.status = "aligning"
        job.progress = 0.6
        job.emit()

        def _align():
            model_a, metadata = get_align_model(detected_lang)
            return whisperx.align(
                result["segments"], model_a, metadata, audio, DEVICE,
                return_char_alignments=False,
            )

        try:
            aligned = await _run_phase(job, _align,
                                       label="Aligning", p_start=0.6, p_end=0.8, eta=eta_al)
            segments = aligned["segments"]
        except Exception as e:
            job.message = f"Align skipped: {e}"
            segments = result["segments"]
        job.progress = 0.8
        job.emit()

        if job.diarize and HF_TOKEN:
            job.status = "diarizing"
            job.emit()

            def _diarize():
                pipeline = get_diarize_pipeline()
                diar_segments = pipeline(str(wav_path))
                merged = whisperx.assign_word_speakers(diar_segments, {"segments": segments})
                return merged["segments"]

            try:
                segments = await _run_phase(job, _diarize,
                                            label="Diarizing", p_start=0.8, p_end=0.99, eta=eta_di)
            except Exception as e:
                job.message = f"Diarize failed: {e}"

        job.progress = 1.0
        job.status = "done"
        job.message = "Done"
        job.result = {
            "language": detected_lang,
            "duration": duration,
            "segments": _clean_segments(segments),
        }
        job.emit()
        _write_srt(job)
    except Exception as e:
        import traceback
        job.status = "error"
        job.error = f"{e}\n\n{traceback.format_exc()}"
        job.message = "Error"
        job.emit()
    finally:
        job.queue.put_nowait(None)
        try:
            if wav_path is not None:
                wav_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            src_path.unlink(missing_ok=True)
            job.src_path = ""
            job.persist()
        except Exception:
            pass


def _clean_segments(segments):
    out = []
    for s in segments:
        out.append({
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", 0.0)),
            "text": (s.get("text") or "").strip(),
            "speaker": s.get("speaker"),
        })
    return out


def _fmt_ts(t: float, sep: str = ",") -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _render_srt(segments) -> str:
    lines = []
    for i, s in enumerate(segments, 1):
        speaker = f"[{s['speaker']}] " if s.get("speaker") else ""
        lines.append(str(i))
        lines.append(f"{_fmt_ts(s['start'])} --> {_fmt_ts(s['end'])}")
        lines.append(f"{speaker}{s['text']}")
        lines.append("")
    return "\n".join(lines)


def _render_vtt(segments) -> str:
    lines = ["WEBVTT", ""]
    for s in segments:
        speaker = f"<v {s['speaker']}>" if s.get("speaker") else ""
        lines.append(f"{_fmt_ts(s['start'], '.')} --> {_fmt_ts(s['end'], '.')}")
        lines.append(f"{speaker}{s['text']}")
        lines.append("")
    return "\n".join(lines)


def _render_txt(segments) -> str:
    out = []
    current_speaker = None
    for s in segments:
        spk = s.get("speaker")
        if spk and spk != current_speaker:
            out.append(f"\n{spk}:")
            current_speaker = spk
        out.append(s["text"])
    return "\n".join(out).strip()


def _render_md(segments, language: str | None, duration: float | None) -> str:
    head = ["# Transcript"]
    if language:
        head.append(f"- language: `{language}`")
    if duration:
        head.append(f"- duration: {int(duration // 60)}m {int(duration % 60)}s")
    head.append("")
    body = []
    current_speaker = None
    for s in segments:
        spk = s.get("speaker")
        ts = f"`{_fmt_ts(s['start'], '.')[:-4]}`"
        if spk and spk != current_speaker:
            body.append(f"\n### {spk}\n")
            current_speaker = spk
        body.append(f"- {ts} {s['text']}")
    return "\n".join(head + body)


def _write_srt(job: Job):
    if not job.result:
        return
    srt = _render_srt(job.result["segments"])
    (DATA_DIR / f"{job.id}.srt").write_text(srt, encoding="utf-8")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_jobs_from_disk()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "model": MODEL_NAME,
        "device": DEVICE,
        "diarize_available": bool(HF_TOKEN),
    }


@app.get("/api/jobs")
async def list_jobs():
    items = []
    for j in JOBS.values():
        items.append({
            "id": j.id,
            "filename": j.filename,
            "status": j.status,
            "progress": j.progress,
            "language": j.language,
            "duration": (j.result or {}).get("duration"),
        })
    items.sort(key=lambda x: x["id"], reverse=True)
    return items


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    diarize: bool = Form(DIARIZE_DEFAULT),
    language: Optional[str] = Form(None),
):
    job_id = uuid.uuid4().hex
    suffix = Path(file.filename or "upload").suffix or ".bin"
    dst = DATA_DIR / f"{job_id}{suffix}"
    with dst.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    job = Job(
        id=job_id,
        filename=file.filename or dst.name,
        src_path=str(dst),
        diarize=diarize and bool(HF_TOKEN),
        language=language,
    )
    JOBS[job_id] = job
    job.persist()
    asyncio.create_task(run_job(job, dst))
    return {"id": job_id, "diarize": job.diarize}


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")

    async def gen():
        if job.status in ("done", "error"):
            yield {"event": "progress", "data": json.dumps({
                "id": job.id, "status": job.status, "progress": job.progress,
                "message": job.message, "error": job.error,
            })}
            return
        while True:
            item = await job.queue.get()
            if item is None:
                break
            yield {"event": "progress", "data": json.dumps(item)}

    return EventSourceResponse(gen())


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return {
        "id": job.id,
        "filename": job.filename,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "error": job.error,
        "language": job.language,
        "diarize": job.diarize,
        "result": job.result,
    }


@app.get("/api/jobs/{job_id}/srt")
async def get_srt(job_id: str):
    job = JOBS.get(job_id)
    if not job or not job.result:
        raise HTTPException(404, "not ready")
    return JSONResponse({"srt": _render_srt(job.result["segments"])})


@app.get("/api/jobs/{job_id}/export")
async def export_job(job_id: str, format: str = "srt"):
    job = JOBS.get(job_id)
    if not job or not job.result:
        raise HTTPException(404, "not ready")
    segs = job.result["segments"]
    fmt = format.lower()
    if fmt == "srt":
        body = _render_srt(segs)
    elif fmt == "vtt":
        body = _render_vtt(segs)
    elif fmt == "txt":
        body = _render_txt(segs)
    elif fmt == "md":
        body = _render_md(segs, job.result.get("language"), job.result.get("duration"))
    elif fmt == "json":
        body = json.dumps(job.result, ensure_ascii=False, indent=2)
    else:
        raise HTTPException(400, f"unknown format: {format}")
    return JSONResponse({"content": body, "format": fmt})


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    job = JOBS.pop(job_id, None)
    if not job:
        raise HTTPException(404, "job not found")
    for ext in (".json", ".srt"):
        (DATA_DIR / f"{job_id}{ext}").unlink(missing_ok=True)
    if job.src_path:
        Path(job.src_path).unlink(missing_ok=True)
    return {"ok": True}
