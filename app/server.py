"""PlateLock app server.

Talks to ComfyUI only through the Comfy Developer Platform (comfy-sdk):
  - self-hosted ComfyUI behind comfy-api-proxy:  COMFY_BASE_URL=http://127.0.0.1:8189
  - Comfy Cloud:                                  COMFY_API_KEY=...

Two job types:
  fx   - your shot + an effect prompt -> LTX-2.3 edit -> PlateLock comp
  lock - your shot + ANY generated clip (bring your own) -> PlateLock comp
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from presets import PRESETS, build_prompt

ROOT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).resolve().parent / "web"
DATA = Path(os.environ.get("PLATELOCK_DATA", Path(__file__).resolve().parent / "data"))
EXAMPLES = ROOT / "examples" if (ROOT / "examples" / "manifest.json").exists() else ROOT / "docs" / "examples"
WF_FX = ROOT / "workflows" / "platelock_ltx_api.json"
WF_LOCK = ROOT / "workflows" / "platelock_comp_api.json"
MAX_SECONDS = float(os.environ.get("PLATELOCK_MAX_SECONDS", str(97 / 24)))  # LTX: 97 frames

OUTPUTS = ["comp", "raw", "xray", "element", "share"]

app = FastAPI(title="PlateLock")
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()

_client = None


def client():
    global _client
    if _client is None:
        from comfy_sdk import Comfy
        _client = Comfy(timeout=120.0)
    return _client


# ---------------------------------------------------------------------------
# workflow helpers: nodes are addressed by their _meta.title, so the graphs can
# be edited in ComfyUI without touching this file
# ---------------------------------------------------------------------------

def by_title(graph: dict, title: str) -> str:
    for nid, node in graph.items():
        if node.get("_meta", {}).get("title") == title:
            return nid
    raise KeyError(f"workflow has no node titled {title!r}")


def maybe_title(graph: dict, title: str) -> str | None:
    try:
        return by_title(graph, title)
    except KeyError:
        return None


def prep_clip(src: Path, dst: Path, max_seconds: float = MAX_SECONDS, max_h: int = 1080):
    """Trim to max_seconds, cap at 1080p, constant 24 fps, keep audio."""
    vf = f"scale=-2:'min({max_h},ih)':flags=lanczos,fps=24"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-t", f"{max_seconds}",
                    "-vf", vf, "-c:v", "libx264", "-crf", "12", "-preset", "fast",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(dst)], check=True)


def duration(path: Path) -> float:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                   "-of", "csv=p=0", str(path)])
    return float(out.decode().strip())


QUIPS = [
    "Asking the model very nicely not to touch your shot…",
    "Rendering the effect with LTX-2.3…",
    "Finding every pixel the model changed…",
    "Ignoring VAE shimmer. We see you, shimmer.",
    "Stabilising the matte so nothing flickers…",
    "Copying your original pixels back, bit for bit…",
    "Re-graining the effect to match your camera…",
]


NODE_QUIPS = {
    "Prep Plate": "Trimming your shot to LTX's favourite length (8n+1 frames)…",
    "LTX-2.3 (open weights)": "Waking up 22 billion open-weight parameters…",
    "Effect Prompt": "Reading your prompt with Gemma…",
    "IC-LoRA Guide (your shot)": "Showing LTX your shot…",
    "LTX Sampler": "Rendering the effect with LTX-2.3… (the slow, magical part)",
    "Output Gamma": "Decoding frames…",
    "Difference Matte": "Finding every pixel the model changed. Ignoring VAE shimmer.",
    "Temporal Stabilize": "Stabilising the matte so nothing flickers…",
    "Composite onto Plate": "Copying your original pixels back, bit for bit…",
    "Fidelity Report": "Measuring how much of your shot survived…",
    "Before/After Share": "Cutting a 9:16 before/after for your feed…",
}


def update(job_id: str, **kw):
    with LOCK:
        JOBS[job_id].update(kw)


def run_job(job_id: str, kind: str, plate: Path, gen: Path | None, prompt: str | None,
            seed: int, sensitivity: float, element_mode: str):
    out_dir = DATA / job_id
    try:
        from comfy_sdk import OutputReady, Progress  # noqa: F401
        c = client()
        graph = json.loads((WF_FX if kind == "fx" else WF_LOCK).read_text())
        graph = copy.deepcopy(graph)
        wf = c.workflows.from_json(graph)
        wf.set_input(by_title(graph, "Plate (original footage)"), "file", c.assets.from_file(str(plate)))
        if kind == "fx":
            wf.set_input(by_title(graph, "Effect Prompt"), "text", prompt)
            nid = maybe_title(graph, "Seed")
            if nid:
                wf.set_input(nid, "noise_seed", seed)
        else:
            wf.set_input(by_title(graph, "Generated (model output)"), "file", c.assets.from_file(str(gen)))
        wf.set_input(by_title(graph, "Difference Matte"), "sensitivity", sensitivity)
        nid = maybe_title(graph, "Extract Element")
        if nid:
            wf.set_input(nid, "mode", element_mode)

        update(job_id, status="running", message=QUIPS[0])
        job = c.submit(wf)
        update(job_id, comfy_job=job.id)
        t0 = time.time()
        try:
            for ev in job.events():
                name = type(ev).__name__
                if name == "Progress":
                    if ev.nodes_total:
                        frac = (ev.nodes_done or 0) / ev.nodes_total
                    else:
                        frac = ev.value
                    title = graph.get(str(ev.current_node), {}).get("_meta", {}).get("title", "")
                    update(job_id, progress=round(min(frac, 0.98), 3),
                           message=NODE_QUIPS.get(title, JOBS[job_id].get("message")))
                elif name == "StatusChange":
                    update(job_id, comfy_status=ev.status)
        except Exception:
            pass  # events are best-effort; result() below is authoritative
        job.result()

        out_dir.mkdir(parents=True, exist_ok=True)
        files = {}
        for name in OUTPUTS:
            nid = maybe_title(graph, f"Save {name}")
            if not nid:
                continue
            outs = job.get_outputs(nid)
            if outs:
                dst = out_dir / f"{name}.mp4"
                outs[0].to_file(dst)
                files[name] = f"/data/{job_id}/{name}.mp4"
        shutil.copy(plate, out_dir / "original.mp4")
        files["original"] = f"/data/{job_id}/original.mp4"

        report = None
        nid = maybe_title(graph, "Fidelity Report")
        for o in (job.get_outputs(nid) if nid else []):
            try:
                txt = o.to_bytes().decode()
                if txt.strip().startswith("{"):
                    report = json.loads(txt)
            except Exception:
                pass
        if report is None:
            report = local_report(out_dir)
        (out_dir / "report.json").write_text(json.dumps(report, indent=2))
        update(job_id, status="done", progress=1.0, files=files, report=report,
               seconds=round(time.time() - t0, 1), message="Locked. Your shot is safe.")
    except Exception as e:  # surface every failure to the UI
        update(job_id, status="error", message=f"{type(e).__name__}: {e}")


def local_report(out_dir: Path) -> dict | None:
    """Fallback: measure fidelity from the downloaded files if the server did
    not return the report text (some proxies drop text outputs)."""
    try:
        import sys
        sys.path.insert(0, str(ROOT / "comfyui-platelock"))
        from platelock import core
        from video_io import read_video
        plate = read_video(out_dir / "original.mp4")
        comp = read_video(out_dir / "comp.mp4")
        raw = read_video(out_dir / "raw.mp4")
        n = min(len(plate), len(comp))
        plate, comp = plate[:n], comp[:n]
        diff = (comp - plate).abs().amax(-1)
        matte = (diff > 6 / 255).float()
        rep = core.fidelity_report(plate, raw, comp, matte)
        rep["note"] = "measured on encoded files"
        return {k: (str(v) if v == float("inf") else v) for k, v in rep.items()}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.get("/api/info")
def info():
    return {"presets": PRESETS, "max_seconds": round(MAX_SECONDS, 1),
            "fx_available": WF_FX.exists(),
            "target": os.environ.get("COMFY_BASE_URL") or "Comfy Cloud"}


@app.post("/api/jobs")
async def create_job(plate: UploadFile = File(...), generated: UploadFile | None = File(None),
                     preset: str = Form("custom"), subject: str = Form(""),
                     seed: int = Form(42), sensitivity: float = Form(0.5),
                     element_mode: str = Form("")):
    kind = "lock" if generated is not None else "fx"
    if kind == "fx" and not WF_FX.exists():
        raise HTTPException(400, "LTX workflow not installed; use bring-your-own mode")
    job_id = uuid.uuid4().hex[:10]
    work = DATA / job_id / "in"
    work.mkdir(parents=True, exist_ok=True)
    raw_plate = work / f"plate_src{Path(plate.filename or 'p.mp4').suffix}"
    raw_plate.write_bytes(await plate.read())
    plate_path = work / "plate.mp4"
    gen_path = None
    max_s = MAX_SECONDS
    if generated is not None:
        gen_path = work / f"generated{Path(generated.filename or 'g.mp4').suffix}"
        gen_path.write_bytes(await generated.read())
        max_s = min(MAX_SECONDS * 4, duration(gen_path))
    prep_clip(raw_plate, plate_path, max_s)

    from presets import BY_ID
    p = BY_ID.get(preset, BY_ID["custom"])
    prompt = build_prompt(preset, subject) if kind == "fx" else None
    mode = element_mode or p["element"]
    JOBS[job_id] = {"id": job_id, "kind": kind, "status": "queued", "progress": 0.0,
                    "preset": preset, "prompt": prompt, "message": "Queued…"}
    threading.Thread(target=run_job, daemon=True,
                     args=(job_id, kind, plate_path, gen_path, prompt, seed, sensitivity, mode)).start()
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404)
    return JOBS[job_id]


@app.get("/api/examples")
def examples():
    mf = EXAMPLES / "manifest.json"
    return JSONResponse(json.loads(mf.read_text()) if mf.exists() else [])


DATA.mkdir(parents=True, exist_ok=True)
app.mount("/data", StaticFiles(directory=DATA), name="data")
if EXAMPLES.exists():
    app.mount("/examples", StaticFiles(directory=EXAMPLES), name="examples")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/", StaticFiles(directory=WEB), name="web")
