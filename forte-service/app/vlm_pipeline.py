import base64
import logging
import math
import os
import time
from typing import Optional

import cv2

log = logging.getLogger("pipeline")

# ---- Tunables (can be overridden with environment variables) -----------------
# Total time one video may take, in seconds (the 6-minute timer). It includes model loading when the
# model was not preloaded, and the evidence-frame extraction at the end (see SAFETY_MARGIN_SEC).
BUDGET_SEC = float(os.environ.get("VLM_BUDGET_SEC", "240"))  # ТЗ: весь ответ за 5 минут, включая загрузку
# How many frames the VLM sees at once, and how many seconds they span. Fewer frames = fewer
# image tokens = faster windows (3 frames is a good laptop trade-off; 4 is a bit more precise).
WINDOW_FRAMES = int(os.environ.get("VLM_WINDOW_FRAMES", "3"))
WINDOW_SPAN_SEC = float(os.environ.get("VLM_WINDOW_SPAN", "3"))
# Distance between window starts. 3s with a 3s span = windows touch, no gaps.
STEP_SEC = float(os.environ.get("VLM_STEP", "3"))
# Ignore detections the model is not sure about.
MIN_CONF = float(os.environ.get("VLM_MIN_CONF", "0.6"))
# "other" (violation outside the known list) is riskier for false positives: higher bar, and it must
# be seen in at least two windows no more than OTHER_PAIR_SEC apart.
OTHER_MIN_CONF = float(os.environ.get("VLM_OTHER_MIN_CONF", "0.8"))
OTHER_PAIR_SEC = float(os.environ.get("VLM_OTHER_PAIR_SEC", "15"))
# Cap on windows per video: long videos get a bigger step so they still fit the budget.
MAX_WINDOWS = int(os.environ.get("VLM_MAX_WINDOWS", "60"))
# Frames are downscaled before going to the VLM (the model caps pixels again, see vlm.MAX_PIXELS).
VLM_MAX_W = int(os.environ.get("VLM_MAX_W", "448"))
# When the budget cannot cover every window: this share of the windows we CAN afford goes to the
# most active ones, the rest is spread evenly so quiet stretches are still sampled (0 = even only,
# 1 = activity only). See motion.py.
MOTION_SHARE = float(os.environ.get("VLM_MOTION_SHARE", "0.8"))
# The same violation seen again within this many seconds is treated as one event.
COOLDOWN_SEC = float(os.environ.get("VLM_COOLDOWN", "6"))
# Seconds kept free at the end of the budget for extracting evidence frames.
SAFETY_MARGIN_SEC = 20
# A new window is only started if it is expected to finish before the deadline: its expected
# duration is the moving average times this factor (a window can be slower than average).
WINDOW_TIME_FACTOR = 1.25
# Guess of seconds per window used only to plan the FIRST window (nothing measured yet); it is
# replaced by the real timing right after that window. Too high/low only affects the order.
PRIOR_WINDOW_SEC = float(os.environ.get("VLM_PRIOR_SEC", "30"))
# JPEG quality of the annotated live-view frames sent to the web page.
TRACE_JPEG_QUALITY = 70


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


def _to_pil(frame):
    """BGR OpenCV frame -> downscaled RGB PIL image (the model expects RGB)."""
    from PIL import Image  # local import: Pillow is only needed for the ML path
    h, w = frame.shape[:2]
    if w > VLM_MAX_W:
        frame = cv2.resize(frame, (VLM_MAX_W, int(h * VLM_MAX_W / w)))
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def _read_window(cap, times: list, fps: float) -> list:
    """
    Read the frames at `times` (ascending seconds) and return [(time, PIL image), ...].
    Seeking in compressed video is slow (the decoder restarts from a keyframe), so we seek
    ONCE to the first time and then move forward by decoding-and-discarding frames with grab().
    """
    out, pos = [], None  # pos = time of the frame we read last
    for tt in times:
        skip = int(round((tt - pos) * fps)) - 1 if pos is not None else 0
        if pos is None or skip > 2 * fps:  # first frame, or a long jump: seeking is cheaper
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, tt) * 1000)
        else:
            for _ in range(max(0, skip)):
                cap.grab()
        ok, frame = cap.read()
        if not ok:
            break
        out.append((tt, _to_pil(frame)))
        pos = tt
    return out


def _window_times(start: float, duration: float) -> list:
    """Evenly spaced timestamps inside [start, start + WINDOW_SPAN_SEC], clamped to the video."""
    end = min(start + WINDOW_SPAN_SEC, max(duration - 0.1, start))
    if WINDOW_FRAMES <= 1:
        return [start]
    gap = (end - start) / (WINDOW_FRAMES - 1)
    return [start + i * gap for i in range(WINDOW_FRAMES)]


def _hex_to_bgr(h: str):
    """'#rrggbb' -> (b, g, r), the colour order OpenCV drawing functions expect."""
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def _annotated_b64(pil_im, people: list, violation_key: Optional[str]) -> str:
    """
    Draw what the model found onto the reference frame and return it as a base64 JPEG:
    a coloured box + "P1 cashier" label per person, and a red border if a violation was flagged.
    """
    import numpy as np
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    h, w = frame.shape[:2]
    for p in people:
        x1, y1, x2, y2 = p["bbox"]  # normalised 0..1 -> pixels of this image
        pt1, pt2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
        color = _hex_to_bgr(p["color"])
        cv2.rectangle(frame, pt1, pt2, color, 2)
        label = f'{p["id"]} {p.get("role", "")}'
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        ly = max(pt1[1], th + 6)  # keep the label inside the image at the top edge
        cv2.rectangle(frame, (pt1[0], ly - th - 6), (pt1[0] + tw + 6, ly), color, -1)
        cv2.putText(frame, label, (pt1[0] + 3, ly - 4), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    if violation_key:  # red frame + rule key so a flagged window is obvious at a glance
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
        cv2.putText(frame, "VIOLATION: " + violation_key, (8, h - 10), font, 0.55,
                    (0, 0, 255), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, TRACE_JPEG_QUALITY])
    return base64.b64encode(buf).decode() if ok else ""


def detect_events(video_path: str, deadline: float, on_progress=None, on_observation=None) -> list:
    """
    Ask the VLM about windows of the video and return events:
      {"type", "timecode_sec", "description", "frame_times", "rule"}
    which analyze_video() turns into violations with evidence frames.

    Which windows, and in what order: a cheap OpenCV pass (motion.py) scores every candidate
    window by how much happens in it. After EVERY window we measure how long a window takes and
    work out how many more fit before the deadline; if not all of them do, the VLM gets the most
    active windows plus an evenly spread coverage set (motion.select_windows) and the rest are
    skipped. The chosen windows are always processed in time order, so the person tracker keeps
    P1/P2 stable. A slow machine therefore analyses fewer windows, but the most promising ones,
    and never runs past the deadline.

    Live view hooks (both optional):
      on_progress(info)    - called at every STAGE (loading model / reading frames / model
                             thinking / window done) with window counters.
      on_observation(obs)  - called once per window with everything the model saw: people
                             (id, role, action, box), verdict, raw answer, annotated frame.
                             The last call is a {"summary": {...}} record about coverage.
    """
    # Imported lazily so the API still starts fast (and tests run) without torch installed.
    from . import vlm
    from .motion import activity_profile, select_windows, window_activity
    from .rules import RULES
    from .tracker import PeopleTracker

    def progress(**info):
        if on_progress:
            on_progress(info)

    stop_at = deadline - SAFETY_MARGIN_SEC

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = n_frames / fps
    if duration <= 0:
        cap.release()
        return []

    # Candidate windows: a regular grid (the densest sampling we would ever use)...
    base_step = max(STEP_SEC, duration / MAX_WINDOWS)
    starts = [i * base_step for i in range(int(math.ceil(duration / base_step)))]
    if len(starts) > 1 and duration - starts[-1] < 1.0:  # a sub-second tail has nothing to judge
        starts.pop()
    # ...each scored by how much moves in it. If the pass fails, every window scores 0 and the
    # planner degrades to evenly spread sampling.
    try:
        t_act = time.time()
        times, scores = activity_profile(video_path)
        activity = window_activity(times, scores, starts, WINDOW_SPAN_SEC)
        log.info("Activity pass: %.1fs for a %.1fs video", time.time() - t_act, duration)
    except Exception:
        log.exception("Activity pass failed - falling back to evenly spread windows")
        activity = [0.0] * len(starts)

    # Load the model BEFORE the loop: if it can't load (no GPU memory, missing
    # weights...) the job fails loudly instead of silently reporting "no_violation".
    # (main.py also preloads it at server start, so normally this returns instantly.)
    progress(stage="loading_model", windows_done=0, windows_total=0, events_found=0)
    vlm.load()

    pending = list(range(len(starts)))  # window indices not processed yet, in time order
    analysed_starts = []                # start times of windows already processed
    other_hits, other_done = [], []  # "other": every confident hit / timecodes already reported
    events, last_seen = [], {}  # last_seen: rule key -> timecode of its last detection
    tracker = PeopleTracker()   # keeps "P1", "P2"... stable between windows
    idx = 0                     # 1-based count of the windows started so far
    ok_windows = 0              # windows the model really analysed (not skipped)
    last_end = 0.0              # end time of the last analysed window
    avg_sec = None              # moving average of seconds per window (reading + model)
    last_started = None
    step = base_step
    log.info("Analyzing %.1fs video: %d candidate windows, budget %.0fs", duration, len(starts),
             stop_at - time.time())

    while pending and time.time() < stop_at:
        now = time.time()
        if last_started is not None:
            # speed of the previous window; the first measurement replaces the prior, after that
            # a moving average forgets the slow first window (warm-up) instead of letting it
            # distort the plan forever
            spent = now - last_started
            avg_sec = spent if avg_sec is None else 0.6 * avg_sec + 0.4 * spent
        # How many more windows fit before the deadline (this one included)?
        est = PRIOR_WINDOW_SEC if avg_sec is None else avg_sec
        affordable = int((stop_at - now) / (est * WINDOW_TIME_FACTOR))
        if avg_sec is None:
            affordable = max(1, affordable)  # always try at least one window
        elif affordable < 1:
            break  # the next window would not finish before the deadline
        last_started = now

        # Pick the windows worth the remaining time, then run the EARLIEST of them
        chosen = select_windows([starts[i] for i in pending], [activity[i] for i in pending],
                                affordable, MOTION_SHARE, covered=analysed_starts)
        pos = min(chosen)
        win = pending.pop(pos)
        idx += 1
        w_start, w_end = starts[win], min(starts[win] + WINDOW_SPAN_SEC, duration)
        analysed_starts.append(w_start)
        planned_total = idx - 1 + len(chosen)
        step = duration / max(1, planned_total)  # effective spacing between analysed windows
        common = dict(window=idx, windows_total=planned_total, window_start=round(w_start, 1),
                      window_end=round(w_end, 1), events_found=len(events), step=round(step, 1))

        progress(stage="reading_frames", windows_done=idx - 1, **common)
        t_read = time.time()
        grabbed = _read_window(cap, _window_times(w_start, duration), fps)
        read_sec = time.time() - t_read
        if len(grabbed) < 2:  # not enough frames to judge an action
            continue

        progress(stage="model_thinking", windows_done=idx - 1, **common)
        t0 = time.time()
        try:
            res = vlm.analyze_window([im for _, im in grabbed])
        except Exception as e:
            # One bad window (e.g. transient OOM) must not kill the whole job,
            # but it is shown in the live view so the reviewer knows it was skipped.
            log.exception("VLM failed on window at %.1fs", w_start)
            if on_observation:
                on_observation({"n": idx, "total": planned_total, "t_start": grabbed[0][0],
                                "t_end": grabbed[-1][0], "error": f"{type(e).__name__}: {e}"[:300],
                                "people": [], "flagged": False})
            continue
        latency = time.time() - t0
        ok_windows += 1
        last_end = max(last_end, grabbed[-1][0])

        key = str(res.get("violation", "none"))
        try:
            conf = float(res.get("confidence", 0))
        except (TypeError, ValueError):
            conf = 0.0
        # A detection counts only for known rules with enough confidence (the model can invent keys)
        if key == "other":
            flagged = conf >= OTHER_MIN_CONF and bool(str(res.get("description", "")).strip())
        else:
            flagged = key in RULES and conf >= MIN_CONF

        # Give people stable ids and colours (P1, P2...) by matching them with earlier windows
        people = tracker.update(res.get("people") or [])

        # Console view of what the model "thinks" for this window + where the time went
        log.info("[%d/%d] %.1fs-%.1fs -> %s (conf %.2f)%s | model %.1fs, reading %.1fs, activity %.2f | %s",
                 idx, planned_total, grabbed[0][0], grabbed[-1][0], key, conf,
                 " FLAGGED" if flagged else "", latency, read_sec, activity[win],
                 str(res.get("description", ""))[:100])
        for p in people:
            log.info("      %s (%s): %s", p["id"], p["role"], p["action"])

        # Everything the live view needs. Boxes refer to the LAST frame, so that is the picture sent.
        if on_observation:
            on_observation({
                "n": idx, "total": planned_total,
                "t_start": round(grabbed[0][0], 2), "t_end": round(grabbed[-1][0], 2),
                "latency_sec": round(latency, 1), "read_sec": round(read_sec, 1),
                "activity": round(activity[win], 3),
                "people": people,
                "model_verdict": key, "confidence": round(conf, 2), "flagged": flagged,
                "violation_title": RULES[key]["title"] if key in RULES else "",
                "description": str(res.get("description", ""))[:300],
                "raw": str(res.get("raw", ""))[:800],
                "frame_b64": _annotated_b64(grabbed[-1][1], people, key if flagged else None),
            })

        if flagged:
            # The model names its best frame; clamp the index in case it is out of range
            try:
                k = min(max(int(res.get("key_frame", 0)), 0), len(grabbed) - 1)
            except (TypeError, ValueError):
                k = 0
            key_time = grabbed[k][0]

            # "other": windows are visited out of order, so keep every hit and report only when
            # two hits are close in time; the earlier one becomes the event.
            if key == "other":
                other_hits.append((key_time, conf, str(res.get("description", "")).strip(),
                                   [tt for tt, _ in grabbed]))
                dup = any(abs(key_time - t) < COOLDOWN_SEC for t in other_done)
                close = [h for h in other_hits[:-1] if abs(h[0] - key_time) <= OTHER_PAIR_SEC]
                if close and not dup:
                    first = min(close + [other_hits[-1]], key=lambda h: h[0])
                    other_done.append(first[0])
                    events.append({
                        "type": "other",
                        "timecode_sec": first[0],
                        "description": f"{first[2]} (confidence {first[1]:.2f})",
                        "frame_times": [first[0]] + [tt for tt in first[3] if tt != first[0]],
                        "rule": RULES["other"]["title"],
                    })

            # Merge repeats: a violation that lasts 20s must be ONE event, not seven
            elif key in last_seen and key_time - last_seen[key] < COOLDOWN_SEC:
                last_seen[key] = key_time
            else:
                last_seen[key] = key_time
                # Best frame first, so analyze_video()'s "first 3 frames" always includes it
                others = [tt for i, (tt, _) in enumerate(grabbed) if i != k]
                events.append({
                    "type": key,
                    "timecode_sec": key_time,
                    "description": f"{str(res.get('description', '')).strip()} (confidence {conf:.2f})",
                    "frame_times": [key_time] + others,
                    "rule": RULES[key]["title"],
                })

        progress(stage="window_done", windows_done=idx, **{**common, "events_found": len(events)})

    cap.release()

    # Tell the live view (and the log) how much of the video was really checked, so a run that
    # ran out of time is never mistaken for a full clean scan. Windows left in `pending` were
    # dropped by the planner (lowest priority) or never reached before the deadline.
    summary = {"windows_analyzed": ok_windows, "windows_skipped": len(pending),
               "windows_candidate": len(starts), "video_sec": round(duration, 1),
               "analyzed_until_sec": round(last_end, 1), "step_sec": round(step, 1),
               "stopped_early": len(pending) > 0}
    log.info("Done: %s", summary)
    if on_observation:
        on_observation({"summary": summary})
    return events


def analyze_video(video_path: str, budget_sec: float = BUDGET_SEC,
                  on_progress=None, on_observation=None) -> dict:
    events = detect_events(video_path, time.time() + budget_sec, on_progress, on_observation)
    violations = []
    for ev in events:
        times = (ev.get("frame_times") or [ev["timecode_sec"]])[:3]
        frames = [f for f in (frame_b64(video_path, t) for t in times) if f]
        v = {"type": ev["type"],
             "timecode_sec": round(float(ev["timecode_sec"]), 2),
             "description": ev["description"],
             "evidence_frames": frames}
        if ev.get("rule"):
            v["rule"] = ev["rule"]
        violations.append(v)
    return {"verdict": "violation" if violations else "no_violation",
            "violations": violations}
