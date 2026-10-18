"""
motion.py - a cheap, model-free "where is something happening?" pass over the video.

Why it exists: the VLM is by far the slowest step. On a laptop it may only afford a handful of
windows inside the time budget, so WHICH windows it gets decides what it can catch. Instead of
spacing them blindly, we:
  1) measure frame-to-frame activity for the whole video (OpenCV only, ~20x faster than real time
     on one CPU core - a 2-minute clip takes a few seconds),
  2) score every candidate window by its activity,
  3) when the budget cannot cover all windows, keep the most active ones PLUS an evenly spread
     "coverage" set, because some events (e.g. a small figure walking to a safe) barely move
     the pixel count and would be missed by activity alone.

Nothing here knows about the rules or the VLM; it only decides where to look.
"""
import math

import cv2
import numpy as np

SAMPLE_FPS = 4            # activity is measured on ~4 frames per second
SMALL_SIZE = (116, 212)   # frames are shrunk to this (w, h) first - enough to see people move
PIXEL_DELTA = 18          # gray-level change (0..255) that counts as "this pixel moved"


def activity_profile(path: str):
    """
    Returns (times, scores): for each sampled frame after the first, its time in seconds and the
    fraction of pixels that changed since the previous sampled frame (0..1).
    Uses grab() (decode only) on skipped frames, so it is much cheaper than reading every frame.
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    every = max(1, int(round(fps / SAMPLE_FPS)))
    times, scores, prev, i = [], [], None, 0
    while cap.grab():
        if i % every == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            small = cv2.resize(frame, SMALL_SIZE, interpolation=cv2.INTER_AREA)
            gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            if prev is not None:
                scores.append(float((cv2.absdiff(gray, prev) > PIXEL_DELTA).mean()))
                times.append(i / fps)
            prev = gray
        i += 1
    cap.release()
    return np.array(times), np.array(scores)


def window_activity(times, scores, starts, span: float) -> list:
    """Total activity inside [start, start + span] for every candidate window start."""
    out = []
    for s in starts:
        m = (times >= s) & (times < s + span)
        out.append(float(scores[m].sum()) if len(times) and m.any() else 0.0)
    return out


def select_windows(starts: list, activity: list, n: int, motion_share: float = 0.8,
                   covered=()) -> set:
    """
    Choose which `n` of the candidate windows to analyse next. Returns a set of indices into `starts`.
      - about `motion_share` of the picks are the most active windows;
      - the rest are chosen to cover the timeline: repeatedly take the window farthest (in time)
        from everything already chosen or already analysed (`covered` = start times of windows
        done earlier), so no long stretch of the video goes completely unseen.
    If everything fits (n >= number of windows) all windows are returned.
    """
    total = len(starts)
    if n >= total:
        return set(range(total))
    if n <= 0:
        return set()
    by_activity = sorted(range(total), key=lambda i: (-activity[i], i))
    chosen = set(by_activity[:max(1, math.ceil(n * motion_share))])
    while len(chosen) < n:
        anchors = [starts[j] for j in chosen] + list(covered)
        best, best_gap = None, -1.0
        for i in range(total):
            if i in chosen:
                continue
            gap = min(abs(starts[i] - a) for a in anchors)
            if gap > best_gap:
                best, best_gap = i, gap
        chosen.add(best)
    return chosen
