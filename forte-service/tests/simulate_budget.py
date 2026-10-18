"""
simulate_budget.py - checks the SCHEDULING and TIME-BUDGET logic of the pipeline without a model.

A stand-in "VLM" replaces the real one: it answers "violation" whenever the window it is shown
contains the labelled second of the video (an oracle), and costs a fixed number of SIMULATED
seconds per window. The pipeline's clock is shifted by those seconds, so a 6-minute budget can be
tested in a few real seconds.

What this tells you:  does the planner reach the labelled moments when the machine is slow, and does
the run stay inside the budget?   What it does NOT tell you: how accurate the real VLM is. For that
use `python -m app.evaluate` on your own laptop.

Usage (from the folder that CONTAINS app/):
    python tests/simulate_budget.py path/to/open [latency_sec ...]
e.g. python tests/simulate_budget.py ../open 2 15 40
Set SIM_ONLY=1,13,38 to simulate only some videos (faster).
"""
import os
import sys
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app import vlm_pipeline as pipeline  # noqa: E402  (needs only OpenCV, no torch)
from app.labels import find_videos, load_labels  # noqa: E402

root = sys.argv[1]
latencies = [float(x) for x in sys.argv[2:]] or [2, 15, 40]
labels = load_labels(os.path.join(root, "open_labels.xlsx"))
videos = find_videos(os.path.join(root, "videos"))
_only = {x.strip() for x in os.environ.get("SIM_ONLY", "").split(",") if x.strip()}
if _only:
    labels = {k: v for k, v in labels.items() if k in _only}

# ---- simulated clock: real time + seconds "spent" by the fake model ------------------------
_real_time = time.time
offset = [0.0]
pipeline.time = types.SimpleNamespace(time=lambda: _real_time() + offset[0])

# ---- stand-in VLM -----------------------------------------------------------------------
state = {"label": None, "latency": 1.0, "times": []}
_orig_read = pipeline._read_window


def _read_spy(cap, times, fps):
    out = _orig_read(cap, times, fps)
    state["times"] = [t for t, _ in out]
    return out


pipeline._read_window = _read_spy


class FakeVLM:
    @staticmethod
    def load():
        return None, None

    @staticmethod
    def analyze_window(frames):
        offset[0] += state["latency"]
        ts, t = state["label"], state["times"]
        hit = ts is not None and t and t[0] - 0.5 <= ts <= t[-1] + 0.5
        k = min(range(len(t)), key=lambda i: abs(t[i] - ts)) if hit else 0
        return {"violation": "phone_use" if hit else "none", "confidence": 0.9,
                "description": "oracle", "key_frame": k, "people": [], "raw": ""}


import app  # noqa: E402
sys.modules["app.vlm"] = FakeVLM
app.vlm = FakeVLM

BUDGET = pipeline.BUDGET_SEC
print(f"budget {BUDGET:.0f}s, safety margin {pipeline.SAFETY_MARGIN_SEC}s, window frames "
      f"{pipeline.WINDOW_FRAMES}, motion share {pipeline.MOTION_SHARE}\n")
print(f"{'latency/window':>14} | {'found':>9} | {'time ok':>7} | {'false+':>6} | {'max used':>9} | "
      f"{'avg windows/video':>17} | {'skipped':>7}")
for lat in latencies:
    state["latency"] = lat
    found = total_v = false_pos = skipped = n_windows = 0
    worst = 0.0
    for num, lab in sorted(labels.items(), key=lambda kv: int(kv[0])):
        state["label"] = lab["time"] if lab["violation"] else None
        obs = []
        start = pipeline.time.time()
        res = pipeline.analyze_video(videos[num], on_observation=obs.append)
        worst = max(worst, pipeline.time.time() - start)
        summ = next(o["summary"] for o in obs if "summary" in o)
        n_windows += summ["windows_analyzed"]
        skipped += summ["windows_skipped"]
        if lab["violation"]:
            total_v += 1
            # a detection counts when its timecode is within 3 s of the labelled second
            found += any(abs(v["timecode_sec"] - lab["time"]) <= 3 for v in res["violations"])
        elif res["violations"]:
            false_pos += 1
    ok = "yes" if worst <= BUDGET else "NO"
    print(f"{lat:>12.0f} s | {found:>3}/{total_v:<3}   | {ok:>7} | {false_pos:>6} | {worst:>7.0f} s | "
          f"{n_windows / len(labels):>17.1f} | {skipped:>7}")
