import json
import logging
import os
import secrets
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import analyze_video

# Show the pipeline's per-window log lines in the server console
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

API_KEY = os.environ.get("API_KEY", "")
DATA = Path(os.environ.get("DATA_DIR", "data"))
UPLOADS, JOBS = DATA / "uploads", DATA / "jobs"
UPLOADS.mkdir(parents=True, exist_ok=True)
JOBS.mkdir(parents=True, exist_ok=True)
MAX_BYTES = int(os.environ.get("MAX_MB", "200")) * 1024 * 1024

executor = ThreadPoolExecutor(max_workers=int(os.environ.get("WORKERS", "1")))
app = FastAPI(title="Cashier discipline video analyzer")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


def check_key(x_api_key: Optional[str] = Header(default=None)):
    if not API_KEY:
        raise HTTPException(500, "API_KEY is not configured on the server")
    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(401, "Invalid or missing X-API-Key")


def save_job(job_id: str, data: dict):
    tmp = JOBS / (job_id + ".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")  # Windows: иначе cp1251 и падение на русском
    tmp.replace(JOBS / (job_id + ".json"))


def trace_path(job_id: str) -> Path:
    """Live-view log of one job: one JSON object per line (what the model saw in each window)."""
    return JOBS / (job_id + ".trace.jsonl")


# ТЗ: ответ за 5 минут считая от загрузки. Задача, простоявшая в очереди, получает меньше времени.
TOTAL_SEC = float(os.environ.get("TOTAL_SEC", "240"))
MIN_SEC = 30.0


def run_job(job_id: str, path: Path, received: float):
    trace = trace_path(job_id)
    trace.write_text("", encoding="utf-8")  # start every job with an empty trace

    # Append-only: a reader polling the file never sees a half-rewritten document
    def observation(obs: dict):
        with trace.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obs, ensure_ascii=False) + "\n")

    try:
        # Every finished window rewrites the job file, so GET /jobs/{id} shows live progress
        def progress(info: dict):
            save_job(job_id, {"job_id": job_id, "status": "processing", "progress": info})

        deadline = max(received + TOTAL_SEC, time.time() + MIN_SEC)
        result = analyze_video(str(path), deadline=deadline,
                               on_progress=progress, on_observation=observation)
        save_job(job_id, {"job_id": job_id, "status": "done", **result})
    except Exception as e:
        logging.getLogger("main").exception("job %s failed", job_id)
        save_job(job_id, {"job_id": job_id, "status": "failed", "error": str(e)})
    finally:
        path.unlink(missing_ok=True)  # ТЗ: тестовые видео не храним


@app.on_event("startup")
def preload_model():
    """
    Load the VLM right after the server starts (in the same single worker that runs jobs).
    Loading takes minutes; doing it here means the first video does not lose that time
    from its own 6-minute budget (see VLM_BUDGET_SEC in pipeline.py). Failure is only logged - the first job retries and reports it.
    """
    def _load():
        log = logging.getLogger("pipeline")
        try:
            from . import pipeline
            pipeline.warmup()
        except Exception:
            log.exception("model preload failed - the first job will retry and show the error")
    executor.submit(_load)


@app.post("/api/v1/analyze", dependencies=[Depends(check_key)])
async def analyze(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".mov"):
        raise HTTPException(400, "Only .mov files are accepted")
    received = time.time()
    job_id = uuid.uuid4().hex[:8]
    path = UPLOADS / (job_id + ".mov")
    size = 0
    with path.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_BYTES:
                out.close()
                path.unlink(missing_ok=True)
                raise HTTPException(413, "File too large")
            out.write(chunk)
    save_job(job_id, {"job_id": job_id, "status": "processing"})
    executor.submit(run_job, job_id, path, received)
    return {"job_id": job_id}


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(check_key)])
def get_job(job_id: str):
    if not job_id.isalnum():
        raise HTTPException(400, "Bad job_id")
    p = JOBS / (job_id + ".json")
    if not p.exists():
        raise HTTPException(404, "Job not found")
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/api/v1/jobs/{job_id}/trace", dependencies=[Depends(check_key)])
def get_trace(job_id: str, since: int = 0):
    """Live view: returns the windows analysed after index `since`, plus the next index to ask for."""
    if not job_id.isalnum():
        raise HTTPException(400, "Bad job_id")
    since = max(0, since)
    p = trace_path(job_id)
    if not p.exists():
        return {"items": [], "next": since}
    items = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        if i < since:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            break  # the last line is still being written - pick it up on the next poll
    return {"items": items, "next": since + len(items)}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
