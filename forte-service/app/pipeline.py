import base64
import logging
import os
import time
import traceback
from typing import Optional

import cv2

from app.rules import RULES, rule_text

log = logging.getLogger("pipeline")


def frame_b64(path: str, t_sec: float, max_w: int = 960) -> Optional[dict]:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_sec) * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    h, w = frame.shape[:2]
    if w > max_w:
        frame = cv2.resize(frame, (max_w, int(h * max_w / w)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None
    return {"timestamp_sec": round(t_sec, 2),
            "image_base64": base64.b64encode(buf).decode()}


def fake_events() -> list:
    # Только для проверки интерфейса: включается FAKE_EVENTS=1
    return [{"kind": "valuables_left", "timecode_sec": 3.0, "frame_times": [3.0]}]


def _torch_available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None
    except Exception:
        return False


def _cuda_available() -> bool:
    if not _torch_available():
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


# auto: модель через transformers только при NVIDIA GPU; иначе (CPU, AMD, Mac) — Ollama, она быстрее
ENGINE = os.environ.get("ML_ENGINE", "auto").lower()   # auto | vlm | ollama | off


def engine() -> str:
    if ENGINE != "auto":
        return ENGINE
    return "vlm" if _cuda_available() else "ollama"


def warmup():
    """Загрузить модель при старте сервера, чтобы первый ролик не тратил на это свой бюджет."""
    e = engine()
    if os.environ.get("FAKE_EVENTS") == "1" or e == "off":
        return
    if e == "vlm":
        from app import vlm
        vlm.load()
        log.info("VLM (transformers) preloaded")
    elif e == "ollama":
        from app import ml
        ml.warmup()


def detect_events(video_path: str, deadline: float, on_progress=None, on_observation=None) -> list:
    """
    Возвращает список событий: {"kind": ключ из app/rules.py, "timecode_sec", "frame_times", "model_note"}.
    Движок: vlm — Qwen3-VL через transformers (app/vlm_pipeline.py); ollama — Qwen3-VL через Ollama (app/ml.py).
    """
    if os.environ.get("FAKE_EVENTS") == "1":
        return fake_events()
    e = engine()
    log.info("engine: %s", e)
    if e == "vlm":
        from app import vlm_pipeline
        raw = vlm_pipeline.detect_events(video_path, deadline, on_progress, on_observation)
        events = []
        for ev in raw:
            kind = ev.get("type")
            if kind not in RULES:          # модель придумала ключ — пропускаем
                continue
            events.append({"kind": kind, "timecode_sec": ev["timecode_sec"],
                           "frame_times": ev.get("frame_times"),
                           "model_note": str(ev.get("description", "")).strip()})
        return events
    if e == "ollama":
        from app import ml
        return ml.detect(video_path, deadline, on_progress=on_progress)
    return []


def build_violation(ev: dict, video_path: str) -> dict:
    info = RULES.get(ev.get("kind"), {})
    vtype = ev.get("type") or info.get("type") or "Возможное нарушение"
    desc = ev.get("description") or info.get("description") or ""
    if ev.get("kind") == "other" and ev.get("model_note"):
        # нового типа нет в справочнике — показываем, что именно увидела модель
        desc = "Замечено: " + ev["model_note"] + ". " + desc
    if ev.get("details"):
        desc = (desc + " " + ev["details"]).strip()
    rule = ev.get("rule") or info.get("rule")
    norm = rule_text(rule)
    if norm:
        desc = desc + " Норматив: " + norm + "."

    t = round(float(ev["timecode_sec"]), 2)
    times = (ev.get("frame_times") or [t])[:3]
    frames = [f for f in (frame_b64(video_path, x) for x in times) if f]

    v = {"type": vtype, "timecode_sec": t, "description": desc, "evidence_frames": frames}
    if ev.get("kind"):
        v["kind"] = ev["kind"]
    if ev.get("model_note"):
        v["model_note"] = ev["model_note"]  # служебное: что «увидела» модель (на английском), для отладки  # для справочника на сайте; лишнее поле схеме ТЗ не мешает
    if rule:
        v["rule"] = rule
    return v


def analyze_video(video_path: str, budget_sec: float = 240, deadline: Optional[float] = None,
                  on_progress=None, on_observation=None) -> dict:
    started = time.time()
    if deadline is None:
        deadline = started + budget_sec
    try:
        # 10 с оставляем на вырезку кадров-доказательств
        events = detect_events(video_path, deadline - 10, on_progress, on_observation)
    except Exception:
        log.error("detect_events failed:\n%s", traceback.format_exc())
        events = []

    violations = []
    for ev in events:
        try:
            v = build_violation(ev, video_path)
            if not v["evidence_frames"]:  # ТЗ: 1–3 кадра; конец ролика может не читаться — берём чуть раньше
                f = frame_b64(video_path, max(0.0, v["timecode_sec"] - 1.0))
                if f:
                    v["evidence_frames"] = [f]
            violations.append(v)
        except Exception:
            log.error("bad event %r:\n%s", ev, traceback.format_exc())

    violations.sort(key=lambda v: v["timecode_sec"])
    log.info("analyze_video: %d violations in %.1f s", len(violations), time.time() - started)
    return {"verdict": "violation" if violations else "no_violation",
            "violations": violations}
