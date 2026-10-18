"""
tracker.py - gives every person a stable label (P1, P2, ...) across analysis windows.

The VLM numbers people from scratch in every window, so "person 1" in one window may be a
different human in the next. We fix that by matching boxes between consecutive windows by
position (overlap / distance) and role. NO faces and no biometrics are used - only where a
person is and what role the model gave them - so nobody is identified, only followed.
"""
import math

# One colour per person, shared by the boxes drawn on the frame and the web page legend
PALETTE = ["#ff5a5a", "#5ef2ff", "#ffd36b", "#7cff6b", "#d58bff", "#ff9a2a", "#6ba8ff", "#ff7ad9"]


def _iou(a, b):
    """Intersection-over-union of two [x1,y1,x2,y2] boxes: 0 = no overlap, 1 = identical."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


class PeopleTracker:
    def __init__(self, min_iou=0.1, max_dist=0.15, max_missed=3):
        self.min_iou = min_iou        # boxes overlapping at least this much may be the same person
        self.max_dist = max_dist      # ...or whose centres are this close (normalised 0..1 units)
        self.max_missed = max_missed  # forget a person unseen for this many windows
        self.tracks = {}              # numeric id -> {"bbox", "role", "missed"}
        self.next_id = 1

    def update(self, people):
        """
        people: [{"bbox": [x1,y1,x2,y2] normalised 0..1, "role": str, "action": str}, ...]
        Returns the same list with "id" ("P1"...) and "color" added to every person.
        """
        # 1) score every (detected person, known track) pair that is plausibly the same human
        pairs = []
        for pi, p in enumerate(people):
            for tid, tr in self.tracks.items():
                iou = _iou(p["bbox"], tr["bbox"])
                (cx, cy), (tx, ty) = _center(p["bbox"]), _center(tr["bbox"])
                dist = math.hypot(cx - tx, cy - ty)
                if iou >= self.min_iou or dist <= self.max_dist:
                    # same role is a strong hint; a smaller distance is a bonus
                    score = iou + (0.2 if p.get("role") == tr["role"] else 0.0) - 0.5 * dist
                    pairs.append((score, pi, tid))

        # 2) greedy matching: best-scoring pairs first, each person/track used once
        pairs.sort(reverse=True)
        assign, used_tracks = {}, set()
        for _, pi, tid in pairs:
            if pi not in assign and tid not in used_tracks:
                assign[pi] = tid
                used_tracks.add(tid)

        # 3) unmatched detections become new people
        for pi in range(len(people)):
            if pi not in assign:
                assign[pi] = self.next_id
                self.next_id += 1

        # 4) age tracks that were not seen this time, drop the ones gone for too long
        for tid in list(self.tracks):
            if tid not in assign.values():
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed:
                    del self.tracks[tid]

        # 5) refresh matched/new tracks and build the output
        out = []
        for pi, p in enumerate(people):
            tid = assign[pi]
            self.tracks[tid] = {"bbox": p["bbox"], "role": p.get("role", "other"), "missed": 0}
            out.append({**p, "id": f"P{tid}", "color": PALETTE[(tid - 1) % len(PALETTE)]})
        return out
