"""
run_local.py - run the analyzer on one video from the terminal and watch it work.

Usage (from the folder that CONTAINS the package, e.g. if the package is `app/`):
    python -m app.run_local path/to/video.mov [out_dir]
Each window the model checks is printed live (time range, verdict, people and their actions).
If out_dir is given, the annotated frame of every window (boxes + P1/P2 labels) is saved there
as a jpg - the quickest way to check that the boxes sit on the right people.
"""
import base64
import json
import logging
import os
import sys

from .pipeline import analyze_video

# Print the pipeline's log lines (window verdicts and per-person actions) to the terminal
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

out_dir = sys.argv[2] if len(sys.argv) > 2 else None
if out_dir:
    os.makedirs(out_dir, exist_ok=True)


def save_frame(obs):
    """Write the annotated frame of one window to out_dir (skipped for failed windows)."""
    if out_dir and obs.get("frame_b64"):
        name = os.path.join(out_dir, "window_%03d.jpg" % obs["n"])
        with open(name, "wb") as f:
            f.write(base64.b64decode(obs["frame_b64"]))


result = analyze_video(sys.argv[1], on_progress=lambda i: None, on_observation=save_frame)

# Do not dump base64 images into the terminal - just report how many were captured
for v in result["violations"]:
    v["evidence_frames"] = f"<{len(v['evidence_frames'])} frames>"
print(json.dumps(result, indent=2, ensure_ascii=False))
