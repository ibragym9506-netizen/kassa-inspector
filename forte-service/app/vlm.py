"""
vlm.py - local Vision-Language Model wrapper (no cloud calls), tuned for laptops.

The model gets a short WINDOW of frames (a few seconds of video) and the list of rules from
rules.py. It answers with strict JSON: a verdict (which rule, if any, is violated) and - when
enabled - every visible person (box, role, action). Several frames let it judge actions
(handing over money, reaching across the counter), not just still objects.

What makes it light enough for a medium laptop (every knob can be overridden by an env variable):
  * a small model: Qwen2.5-VL-3B instead of 7B (about 7 GB in bf16 instead of about 16 GB);
  * a hard cap on image tokens per frame (VLM_MAX_PIXELS), the main driver of prefill time;
  * a short answer: the VERDICT COMES FIRST, then the (optional) people list. Generating tokens
    is the slowest part on CPU, so the answer is kept small and, if it is ever cut off, the
    verdict has already been written and is still recovered by _parse_json();
  * the people list (boxes) is switched off by default on CPU-only machines (VLM_PEOPLE=1 to force).

One-time setup (shell, NOT python - run once while online, afterwards it works offline):
    huggingface-cli download Qwen/Qwen3-VL-2B-Instruct

LICENCE: check it for the exact checkpoint you ship. Qwen2.5-VL-7B-Instruct is Apache-2.0; the 3B
checkpoint is published under the Qwen *research* licence, which may not allow commercial use.
If you need a guaranteed-commercial model, set VLM_MODEL to an Apache-2.0 checkpoint you have
validated (e.g. Qwen/Qwen2.5-VL-7B-Instruct - heavier). Box parsing below assumes Qwen2.5-VL's
pixel coordinates; with another model family keep VLM_PEOPLE=0 (the verdict still works).
"""
import json
import os
import re
import logging

import torch
from transformers import AutoProcessor

# Relative import: this file lives in the same package as main.py / pipeline.py
from .rules import rules_for_prompt

log = logging.getLogger("pipeline")

# Qwen2.5-VL-3B — лицензия Qwen Research (только некоммерческое использование) — по ТЗ НЕЛЬЗЯ.
# Qwen3-VL-2B-Instruct — Apache 2.0 (коммерция разрешена) и помещается в 4 ГБ видеопамяти.
MODEL_NAME = os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
PATCH = 14  # Qwen2.5-VL cuts images into 14x14 pixel patches; the grid size tells us the real input size

# Upper bound on pixels per frame as seen by the model. Qwen2.5-VL spends ONE token per 28x28
# pixels, so the default 256*28*28 = at most 256 image tokens per frame (a 3-frame window is
# then about 770 image tokens). Raise it if small objects (phones, banknotes) are missed.
MAX_PIXELS = int(os.environ.get("VLM_MAX_PIXELS", str(256 * 28 * 28)))
MIN_PIXELS = int(os.environ.get("VLM_MIN_PIXELS", str(64 * 28 * 28)))

_model = None
_processor = None
_people_on = True  # resolved in load(): people/boxes need extra output tokens


def _device() -> str:
    forced = os.environ.get("VLM_DEVICE")  # "cuda" | "mps" | "cpu"
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype(device: str):
    forced = os.environ.get("VLM_DTYPE")  # e.g. "float32" (faster on CPUs without bf16, needs ~2x RAM)
    if forced:
        return getattr(torch, forced)
    if device == "cuda" and not torch.cuda.is_bf16_supported():
        return torch.float16
    # bfloat16 keeps the 3B model at about 7 GB, which is what lets it fit a 16 GB laptop
    return torch.bfloat16


def _model_class():
    if "qwen2.5-vl" in MODEL_NAME.lower() or "qwen2_5" in MODEL_NAME.lower():
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration
    try:
        from transformers import AutoModelForImageTextToText as auto
    except ImportError:  # older transformers
        from transformers import AutoModelForVision2Seq as auto
    return auto


def load():
    """Load weights once (lazy singleton). After the first download it works offline."""
    global _model, _processor, _people_on
    if _model is None:
        device = _device()
        dtype = _dtype(device)
        attn = "sdpa"  # PyTorch's built-in fast attention; works on every device
        if device == "cuda":
            try:
                import flash_attn  # noqa: F401
                attn = "flash_attention_2"
            except ImportError:
                pass
        kwargs = dict(torch_dtype=dtype, attn_implementation=attn, low_cpu_mem_usage=True)
        if device == "cuda":
            kwargs["device_map"] = "auto"  # spills layers to RAM if the laptop GPU is small
        model = _model_class().from_pretrained(MODEL_NAME, **kwargs)
        if device != "cuda":
            model = model.to(device)
        _model = model.eval()
        _processor = AutoProcessor.from_pretrained(
            MODEL_NAME, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)

        people = os.environ.get("VLM_PEOPLE", "auto").lower()
        _people_on = (device != "cpu") if people == "auto" else people in ("1", "true", "yes", "on")
        log.info("VLM ready: %s on %s (%s), max %d px/frame, people boxes %s",
                 MODEL_NAME, device, str(dtype).replace("torch.", ""), MAX_PIXELS,
                 "ON" if _people_on else "OFF (set VLM_PEOPLE=1 to enable)")
    return _model, _processor


_HEAD = """You are an assistant of a bank security reviewer. The {n} images are consecutive
frames (static camera, looking down at a cashier workplace) in time order.
{task_people}Check ONLY for these visually observable violations:
{rules}

Rules for your answer:
- Report a violation only if you can SEE it. Do not guess about hidden data (postings, journals).
- Do not identify anyone by face.
- Answer with JSON only, in exactly this key order:
"""

_TASK_PEOPLE = """List every person visible in the LAST image (at most 3): pixel box [x1, y1, x2, y2]
in the LAST image, role ("cashier", "customer" or "other") and what they do in 2-4 words.
"""

_SCHEMA_PEOPLE = ('{{"violation": "<rule key or none>", "confidence": 0..1, '
                  '"description": "one short sentence if there is a violation, otherwise empty", '
                  '"key_frame": <0-based index of the best frame>, '
                  '"people": [{{"bbox": [x1, y1, x2, y2], "role": "...", "action": "..."}}]}}')

_SCHEMA_PLAIN = ('{{"violation": "<rule key or none>", "confidence": 0..1, '
                 '"description": "one short sentence if there is a violation, otherwise empty", '
                 '"key_frame": <0-based index of the best frame>}}')


def build_prompt(n_frames: int, people: bool) -> str:
    """The rules always come from rules.py, so edits there change what the model looks for."""
    head = _HEAD.format(n=n_frames, rules=rules_for_prompt(),
                        task_people=_TASK_PEOPLE if people else "")
    return head + (_SCHEMA_PEOPLE if people else _SCHEMA_PLAIN).format()


def analyze_window(pil_frames):
    """Run the VLM on a list of PIL images and return a parsed dict (incl. \"people\" and \"raw\")."""
    model, processor = load()
    content = [{"type": "image", "image": im} for im in pil_frames]
    content.append({"type": "text", "text": build_prompt(len(pil_frames), _people_on)})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=pil_frames, return_tensors="pt").to(model.device)

    # Token budget: verdict about 50 tokens; each person (box + role + action) about 40 more.
    default_budget = "230" if _people_on else "90"
    max_new = int(os.environ.get("VLM_MAX_NEW_TOKENS", default_budget))
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    # Cut the prompt tokens off, keep only the newly generated answer
    answer = processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                    skip_special_tokens=True)[0]

    res = _parse_json(answer)
    res["raw"] = answer  # kept so the web page can show exactly what the model said

    # The model reports boxes in pixels of the image AS IT SAW IT (resized to a multiple of 28).
    # image_grid_thw holds that size in patches, so multiply by the patch size to get pixels.
    grid = inputs.get("image_grid_thw")
    if grid is not None:
        rh, rw = int(grid[-1][1]) * PATCH, int(grid[-1][2]) * PATCH
    else:  # fallback: assume it saw the image unresized
        rw, rh = pil_frames[-1].size
    res["people"] = _normalize_people(res.get("people"), rw, rh) if _people_on else []
    return res


def _normalize_people(raw_people, rw, rh):
    """Validate people from the model and convert boxes to 0..1 so they work at any image size."""
    people = []
    for p in raw_people if isinstance(raw_people, list) else []:
        try:
            x1, y1, x2, y2 = [float(v) for v in p["bbox"]]
        except (KeyError, TypeError, ValueError):
            continue  # box missing or malformed -> skip this person
        if max(x1, y1, x2, y2) > 1.5:  # pixels -> 0..1 (values <= 1.5 are already normalised)
            x1, x2, y1, y2 = x1 / rw, x2 / rw, y1 / rh, y2 / rh
        x1, y1, x2, y2 = [min(1.0, max(0.0, v)) for v in (x1, y1, x2, y2)]
        if x2 - x1 < 0.01 or y2 - y1 < 0.01:  # empty / inverted box
            continue
        role = str(p.get("role", "other")).lower()
        people.append({"bbox": [x1, y1, x2, y2],
                       "role": role if role in ("cashier", "customer", "other") else "other",
                       "action": str(p.get("action", "")).strip()[:80]})
    return people


def _parse_json(s):
    """The model may wrap JSON in text/markdown, or be cut off mid-answer; recover what we can."""
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return _salvage(s)


def _salvage(s):
    """
    Truncated / slightly malformed JSON: pull the verdict fields out with regexes and keep every
    COMPLETE person object. The verdict is written first, so a cut-off answer still has it.
    """
    res = {"violation": "none", "confidence": 0.0, "people": []}
    m = re.search(r'"violation"\s*:\s*"([^"]*)"', s)
    if m:
        res["violation"] = m.group(1)
    m = re.search(r'"confidence"\s*:\s*(\d*\.?\d+)', s)
    if m:
        res["confidence"] = float(m.group(1))
    m = re.search(r'"description"\s*:\s*"([^"]*)"', s)
    if m:
        res["description"] = m.group(1)
    m = re.search(r'"key_frame"\s*:\s*(\d+)', s)
    if m:
        res["key_frame"] = int(m.group(1))
    for pm in re.finditer(r'\{[^{}]*"bbox"[^{}]*\}', s):
        try:
            res["people"].append(json.loads(pm.group(0)))
        except json.JSONDecodeError:
            continue
    return res
