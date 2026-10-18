"""
labels.py - reads open_labels.xlsx (columns: file name | timestamp of the violation | verdict).

The sheet is in Russian: the timestamp looks like "22 секунда" or "1минута 8 секунда", the verdict
is "нарушение" (violation) or "без нарушения" (no violation). Videos are matched to rows by the
NUMBER in the name only (X-1, Х-3 with a Cyrillic Х, "#U0425-3.mov" after a bad unzip...).
"""
import os
import re


def parse_timestamp(text):
    """'22 секунда' -> 22; '1минута 8 секунда' -> 68; empty -> None."""
    if not text:
        return None
    t = str(text).lower()
    mins = re.search(r"(\d+)\s*мин", t)
    secs = re.search(r"(\d+)\s*сек", t)
    if not (mins or secs):
        return None
    return (int(mins.group(1)) * 60 if mins else 0) + (int(secs.group(1)) if secs else 0)


def load_labels(xlsx_path):
    """Returns {video number (str): {"violation": bool, "time": seconds or None}}."""
    import openpyxl  # only needed for evaluation, not for the web service
    ws = openpyxl.load_workbook(xlsx_path, read_only=True).active
    labels = {}
    for row in list(ws.iter_rows(values_only=True))[1:]:
        if not row or not row[0]:
            continue
        num = re.findall(r"\d+", str(row[0]))
        if not num:
            continue
        verdict = str(row[2] or "").strip().lower()
        labels[num[-1]] = {"violation": bool(verdict) and not verdict.startswith("без"),
                           "time": parse_timestamp(row[1])}
    return labels


def find_videos(videos_dir):
    """Returns {video number (str): path} for every .mov in the folder."""
    out = {}
    for name in sorted(os.listdir(videos_dir)):
        m = re.search(r"(\d+)\.mov$", name, re.I)
        if m:
            out[m.group(1)] = os.path.join(videos_dir, name)
    return out
