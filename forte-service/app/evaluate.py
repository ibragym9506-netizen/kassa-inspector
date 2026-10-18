"""
evaluate.py - scores the analyzer against the labelled videos (open_labels.xlsx).

Usage (from the folder that CONTAINS the package `app/`):
    python -m app.evaluate path/to/open [--tol 5] [--only 1,2,13] [--out results.csv]

`path/to/open` is the unzipped open.zip: it holds open_labels.xlsx and the videos/ folder.
Each video is analysed exactly like a real upload (same time budget), then compared with its label:
  - video level : did we say "violation" when the label says violation, and vice versa;
  - moment level: for correctly flagged violation videos, was an event within +-tol seconds
                  of the labelled second.
The labels only say WHETHER and WHEN, not WHICH rule - so the rule type is not scored.
Tip: re-run with a different VLM_MIN_CONF (e.g. 0.5 / 0.7) to see how the threshold trades
false alarms for misses; edit rules.py to see what your own rules change.
"""
import argparse
import csv
import hashlib
import logging
import os
import time

from .labels import find_videos, load_labels
from .pipeline import analyze_video


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="folder with open_labels.xlsx and videos/")
    ap.add_argument("--tol", type=float, default=5.0, help="allowed error of the event time, seconds")
    ap.add_argument("--only", default="", help="comma-separated video numbers, e.g. 1,2,13")
    ap.add_argument("--out", default="evaluation.csv")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    labels = load_labels(os.path.join(args.root, "open_labels.xlsx"))
    videos = find_videos(os.path.join(args.root, "videos"))
    nums = [n for n in sorted(labels, key=int) if n in videos]
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        nums = [n for n in nums if n in wanted]

    # Byte-identical files with different labels make a fair score impossible - say so up front.
    by_hash = {}
    for n in nums:
        by_hash.setdefault(_md5(videos[n]), []).append(n)
    for group in by_hash.values():
        if len(group) > 1 and len({(labels[n]["violation"], labels[n]["time"]) for n in group}) > 1:
            print("WARNING: videos", ", ".join("X-" + n for n in group),
                  "are the same file but have different labels:",
                  {"X-" + n: labels[n]["time"] for n in group})

    rows, tp = [], dict(tp=0, fp=0, fn=0, tn=0, timed=0)
    print(f"{'video':>6} {'label':>10} {'predicted':>16} {'time':>7}  verdict")
    for n in nums:
        lab = labels[n]
        t0 = time.time()
        res = analyze_video(videos[n])
        spent = time.time() - t0
        events = res["violations"]
        pred = bool(events)
        near = [e for e in events if lab["time"] is not None and abs(e["timecode_sec"] - lab["time"]) <= args.tol]
        if lab["violation"] and pred:
            tp["tp"] += 1
            tp["timed"] += bool(near)
            verdict = "OK" if near else "flagged, wrong moment"
        elif lab["violation"]:
            tp["fn"] += 1
            verdict = "MISSED"
        elif pred:
            tp["fp"] += 1
            verdict = "FALSE ALARM"
        else:
            tp["tn"] += 1
            verdict = "OK"
        lab_txt = f"{lab['time']}s" if lab["violation"] else "clean"
        pred_txt = ", ".join(f"{e['type']}@{e['timecode_sec']:.0f}s" for e in events[:2]) or "clean"
        print(f"{'X-' + n:>6} {lab_txt:>10} {pred_txt:>16} {spent:>6.0f}s  {verdict}")
        rows.append({"video": "X-" + n, "label_violation": lab["violation"], "label_time": lab["time"],
                     "predicted_violation": pred,
                     "events": "; ".join(f"{e['type']}@{e['timecode_sec']:.1f}" for e in events),
                     "seconds": round(spent, 1), "verdict": verdict})

    total = len(rows)
    prec = tp["tp"] / (tp["tp"] + tp["fp"]) if tp["tp"] + tp["fp"] else 0.0
    rec = tp["tp"] / (tp["tp"] + tp["fn"]) if tp["tp"] + tp["fn"] else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    print(f"\nvideos {total} | video-level accuracy {(tp['tp'] + tp['tn']) / max(1, total):.0%} | "
          f"precision {prec:.0%} recall {rec:.0%} F1 {f1:.2f}")
    print(f"moment hit (within +-{args.tol:g}s) in {tp['timed']}/{tp['tp'] + tp['fn']} violation videos | "
          f"mean {sum(r['seconds'] for r in rows) / max(1, total):.0f}s, "
          f"max {max((r['seconds'] for r in rows), default=0):.0f}s per video")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["video"])
        w.writeheader()
        w.writerows(rows)
    print("details saved to", args.out)


if __name__ == "__main__":
    main()
