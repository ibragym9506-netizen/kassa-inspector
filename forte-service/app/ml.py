"""
ML-модуль (план Б): локальная VLM через Ollama + правила из аналитики датасета.

Как работает:
  1) из видео берутся кадры с шагом ~1-2 с (шаг подстраивается под бюджет времени);
  2) на каждый кадр локальной модели Qwen3-VL задаётся один вопрос-чеклист, ответ в JSON;
  3) по последовательности ответов срабатывают правила; чтобы не было ложных
     срабатываний, признак должен держаться минимум на 2 кадрах подряд.

Всё локально: Ollama слушает только localhost:11434, ключей и облака нет.
Если Ollama не запущена или модель не скачана — возвращается пустой список
(сервис ответит «нарушений нет», но не упадёт).

Настройки через переменные окружения:
  OLLAMA_URL   (по умолчанию http://localhost:11434)
  VLM_MODEL    (по умолчанию qwen3-vl:2b)
  ML_STEP_SEC  (желаемый шаг между кадрами, по умолчанию 1.5)
"""
import base64
import json
import logging
import os
import time
import urllib.request
from typing import List, Optional

import cv2

log = logging.getLogger("ml")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
# instruct-версия: без «рассуждений вслух», в разы быстрее (лицензия Apache 2.0)
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-vl:2b-instruct")
STEP = float(os.environ.get("ML_STEP_SEC", "1.5"))

FIELDS = [
    "cashier_present",      # кассир в кадре, у рабочего места
    "customers_count",      # число клиентов/посетителей (не кассир)
    "cash_on_desk",         # наличные лежат на столе
    "cashier_handling_cash",# кассир держит/пересчитывает деньги, кладёт в машину
    "sealed_bag_on_desk",   # опломбированный пакет на столе
    "papers_or_stamps_on_desk",
    "safe_door_open",       # дверца сейфа открыта
    "room_door_open",       # дверь кассового помещения открыта
    "personal_bag_or_coat_on_desk",
    "drink_or_food_on_desk",
    "screen_shows_open_app",# на экране открыто рабочее приложение (не чёрный экран и не экран блокировки)
    "cashier_using_phone",  # кассир смотрит в телефон / говорит / снимает (просто держит — не считается)
]

PROMPT = """You see one frame from a CCTV camera in a bank cashier room.
Answer ONLY with JSON with exactly these keys:
{
 "cashier_present": true/false,            // the bank cashier (employee) is in the room at the desk
 "customers_count": 0/1/2/3,               // number of OTHER people (customers, visitors) in the room
 "cash_on_desk": true/false,               // banknotes or stacks of money lying on the desk
 "cashier_handling_cash": true/false,      // cashier is holding or counting money right now
 "sealed_bag_on_desk": true/false,         // a sealed plastic bag/package with money on the desk
 "papers_or_stamps_on_desk": true/false,   // a folder with documents or rubber stamps on the desk
 "safe_door_open": true/false,             // the door of the safe is open
 "room_door_open": true/false,             // the door of the room is open
 "personal_bag_or_coat_on_desk": true/false, // handbag, backpack or jacket lying on the work desk
 "drink_or_food_on_desk": true/false,      // cup, glass, bottle or food on the desk
 "screen_shows_open_app": true/false,      // computer screen shows an open program window (false if screen is black/off)
 "cashier_using_phone": true/false         // cashier is looking at, talking on or photographing with a mobile phone (just holding it = false)
}
If unsure about a field, answer false (or 0). No explanations."""


# ---------- кадры ----------

def _video_info(path: str):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return (n / fps) if fps else 0.0


def _frame_jpeg_b64(cap, t: float, width: int = 640) -> Optional[str]:
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    if not ok:
        return None
    h, w = frame.shape[:2]
    if w > width:
        frame = cv2.resize(frame, (width, int(h * width / w)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode() if ok else None


# ---------- модель ----------

def _ask(img_b64: str, timeout: float) -> Optional[dict]:
    body = json.dumps({
        "model": MODEL, "prompt": PROMPT, "images": [img_b64],
        "format": "json", "stream": False, "think": False, "keep_alive": "30m",
        "options": {"temperature": 0},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read().decode())
    try:
        raw = json.loads(resp.get("response", "{}"))
    except ValueError:
        return None
    out = {}
    for k in FIELDS:
        v = raw.get(k)
        if k == "customers_count":
            try:
                out[k] = max(0, min(3, int(v)))
            except (TypeError, ValueError):
                out[k] = 0
        else:
            out[k] = v is True or str(v).lower() == "true"
    return out


def ollama_ready() -> bool:
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=3) as r:
            tags = json.loads(r.read().decode())
        names = [m.get("name", "") for m in tags.get("models", [])]
        return MODEL in names or any(n.split(":")[0] == MODEL.split(":")[0] for n in names)
    except Exception:
        return False


def warmup():
    """Загрузить модель в память заранее (первый запрос к Ollama долгий)."""
    if not ollama_ready():
        log.warning("Ollama/модель %s недоступна — проверьте `ollama pull %s`", MODEL, MODEL)
        return
    body = json.dumps({"model": MODEL, "prompt": "ok", "stream": False, "think": False,
                       "keep_alive": "30m", "options": {"num_predict": 1}}).encode()
    req = urllib.request.Request(OLLAMA_URL + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        r.read()
    log.info("Ollama %s загружена за %.1f с", MODEL, time.time() - t)


# ---------- правила ----------

def _runs(samples: List[dict], cond, min_len: int = 2):
    """Начала участков, где cond(s) истинно min_len кадров подряд."""
    starts, run = [], 0
    for i, s in enumerate(samples):
        if cond(s):
            run += 1
            if run == min_len:
                starts.append(i - min_len + 1)
        else:
            run = 0
    return starts


def _apply_rules(samples: List[dict]) -> List[dict]:
    ev = []

    def add(kind, i):
        t = samples[i]["t"]
        nxt = samples[i + 1]["t"] if i + 1 < len(samples) else t
        ev.append({"kind": kind, "timecode_sec": round(t, 2), "frame_times": [t, nxt]})

    def first(kind, cond, min_len=2):
        r = _runs(samples, cond, min_len)
        if r:
            add(kind, r[0])

    # Ушла, оставив ценности (Х-15, Х-24, Х-26)
    first("valuables_left", lambda s: not s["cashier_present"] and s["customers_count"] == 0 and
          (s["cash_on_desk"] or s["sealed_bag_on_desk"] or s["papers_or_stamps_on_desk"]))
    # Ушла, оставив открытое приложение (Х-11, Х-28). Горящий экран без приложения — норма.
    first("app_left_open", lambda s: not s["cashier_present"] and s["screen_shows_open_app"])
    # Дверь кассы открыта при кассире внутри (Х-30, Х-32, Х-33)
    first("door_open", lambda s: s["cashier_present"] and s["room_door_open"])
    # Сейф открыт во время работы с деньгами (Х-20)
    first("safe_open", lambda s: s["cashier_present"] and s["safe_door_open"] and
          (s["cash_on_desk"] or s["cashier_handling_cash"]))
    # Личные вещи на рабочем столе (Х-40)
    first("personal_items", lambda s: s["personal_bag_or_coat_on_desk"])
    # Еда/напитки рядом с деньгами (Х-46)
    first("food_near_cash", lambda s: s["drink_or_food_on_desk"] and s["cash_on_desk"])
    # Телефон при работе с наличными (просто в руке — не нарушение)
    first("phone_during_cash", lambda s: s["cashier_present"] and s["cashier_using_phone"] and
          (s["cash_on_desk"] or s["cashier_handling_cash"]))
    # Два посетителя одновременно (Х-18)
    first("parallel_service", lambda s: s["customers_count"] >= 2 and s["cashier_present"])

    # Клиент ушёл до конца пересчёта (Х-34): был клиент → клиента нет, кассир на месте, деньги в работе
    for i in range(1, len(samples) - 1):
        a, b, c = samples[i - 1], samples[i], samples[i + 1]
        if a["customers_count"] >= 1 and b["customers_count"] == 0 and c["customers_count"] == 0 \
                and b["cashier_present"] and (b["cashier_handling_cash"] or b["cash_on_desk"]) \
                and (c["cashier_handling_cash"] or c["cash_on_desk"]):
            add("count_without_client", i)
            break

    return ev


# ---------- точка входа ----------

def detect(video_path: str, deadline: float, on_progress=None) -> List[dict]:
    if not ollama_ready():
        log.warning("Ollama/модель %s недоступна — ML пропущен", MODEL)
        return []

    duration = _video_info(video_path)
    if duration <= 0:
        return []

    cap = cv2.VideoCapture(video_path)
    samples, t, step = [], 0.3, STEP
    try:
        while t < duration and time.time() < deadline - 5:
            img = _frame_jpeg_b64(cap, t)
            if img is None:
                break
            t_call = time.time()
            try:
                ans = _ask(img, timeout=max(5.0, deadline - time.time() - 3))
            except Exception as e:
                log.warning("VLM error at %.1fs: %s", t, e)
                ans = None
            if ans is not None:
                ans["t"] = round(t, 2)
                samples.append(ans)
                log.info("ML %.1fs (%.1f с): %s", t, time.time() - t_call,
                         {k: v for k, v in ans.items() if v and k != "t"})
            if on_progress:
                try:
                    on_progress({"stage": "frame_done", "t_sec": round(t, 1),
                                 "video_sec": round(duration, 1), "frames_done": len(samples)})
                except Exception:
                    pass
            # подстраиваем шаг, чтобы уложиться в бюджет
            per_call = time.time() - t_call
            left = deadline - time.time() - 5
            remaining = max(0.0, duration - t)
            if left > 0 and per_call > 0:
                can_do = max(1.0, left / per_call)
                step = max(STEP, remaining / can_do)
            t += step
    finally:
        cap.release()

    log.info("ML: %d кадров проанализировано, шаг ~%.1f с", len(samples), step)
    return _apply_rules(samples)
