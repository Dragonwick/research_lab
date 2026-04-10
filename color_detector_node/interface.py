"""
Color Detector Interface
------------------------
Headless camera loop that continuously updates per-slot color trackers.
Runs in a background thread so the MADSci REST node stays responsive.
All public methods are thread-safe.
"""

import subprocess
import threading
import json
import os
from collections import deque
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import ollama


# ── Detection constants ────────────────────────────────────────────────────────
SLOT_CROP_SIZE  = 100     # normal crop radius for live tracking
ZOOM_CROP_SIZE  = 160     # larger crop for zoomed one-shot scans
ZOOM_OUT_SIZE   = 240     # upscale target for zoomed crops
TRAIN_FRAMES    = 40
HISTORY_LEN     = 30
CONFIRM_THRESH  = 0.55
MAX_COLOR_DIST  = 6000
SAT_CUT         = 50      # min saturation for colorful-pixel filter

HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE))   # ot2files/


def _path(fname):
    return os.path.join(DATA_DIR, fname)


def _load(fname, default):
    try:
        with open(_path(fname)) as f:
            return json.load(f)
    except Exception:
        return default


def _save(fname, data):
    with open(_path(fname), "w") as f:
        json.dump(data, f, indent=2)


# ── HSV helpers ───────────────────────────────────────────────────────────────

def crop_slot(frame, cx, cy, crop_size=SLOT_CROP_SIZE):
    h, w = frame.shape[:2]
    half = crop_size // 2
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(w, cx + half), min(h, cy + half)
    r = frame[y1:y2, x1:x2]
    return r if r.size > 0 else None


def crop_slot_zoomed(frame, cx, cy):
    """Larger crop upscaled to ZOOM_OUT_SIZE for more detailed per-slot analysis."""
    region = crop_slot(frame, cx, cy, crop_size=ZOOM_CROP_SIZE)
    if region is None:
        return None
    return cv2.resize(region, (ZOOM_OUT_SIZE, ZOOM_OUT_SIZE),
                      interpolation=cv2.INTER_LINEAR)


def dominant_hsv(region):
    """Median HSV of only high-saturation pixels. Returns None if slot is empty."""
    if region is None or region.size == 0:
        return None
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = hsv[:, :, 1] > SAT_CUT
    colorful = hsv[mask]
    if len(colorful) < 8:
        return None
    med = np.median(colorful, axis=0)
    return float(med[0]), float(med[1]), float(med[2])


def mean_hsv(region):
    """Plain mean HSV — used for floor baseline (floor is grey, dominant_hsv would fail)."""
    if region is None or region.size == 0:
        return None
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    m = np.mean(hsv.reshape(-1, 3), axis=0)
    return float(m[0]), float(m[1]), float(m[2])


def hsv_dist(h1, s1, v1, h2, s2, v2):
    dh = min(abs(h1 - h2), 180 - abs(h1 - h2))
    return (dh * 2.5) ** 2 + (s1 - s2) ** 2 + ((v1 - v2) * 0.7) ** 2


def raw_detect(hsv, trained):
    """Single-frame nearest-centroid detection. Returns (color, distance)."""
    if hsv is None:
        return "Empty", 0
    if not trained:
        return "Unclear", 0
    h, s, v = hsv
    best, best_d = "Unclear", float("inf")
    for name, (th, ts, tv) in trained.items():
        d = hsv_dist(h, s, v, th, ts, tv)
        if d < best_d:
            best_d, best = d, name
    return (best, best_d) if best_d <= MAX_COLOR_DIST else ("Unclear", best_d)


# Hue-range thresholds (from old color.py — no training needed)
MIN_BRIGHTNESS  = 80
MIN_SATURATION  = 120

def hue_range_detect(region) -> str:
    """
    HSV hue-range detection for vivid lab dyes.
    Uses the median of high-saturation pixels only (ignores grey/silver background).
    Covers Red, Orange, Yellow, Green, Blue, Purple.
    Returns Empty when the slot has no saturated pixels (tip rack, clear plate, silver deck).
    """
    if region is None or region.size == 0:
        return "Empty"
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    # Only consider pixels with enough saturation AND brightness
    mask = (hsv[:, :, 1] > MIN_SATURATION) & (hsv[:, :, 2] > MIN_BRIGHTNESS)
    colorful = hsv[mask]
    if len(colorful) < 200:         # require substantial colorful area → else empty
        return "Empty"
    h_val = float(np.median(colorful[:, 0]))

    # OpenCV hue is 0-180
    if h_val < 10 or h_val > 165:
        return "Red"
    elif 10 <= h_val < 22:
        return "Orange"
    elif 22 <= h_val < 38:
        return "Yellow"
    elif 38 <= h_val < 85:
        return "Green"
    elif 85 <= h_val < 130:
        return "Blue"
    elif 130 <= h_val <= 165:
        return "Purple"
    return "Empty"


VALID_COLORS = {"Red", "Yellow", "Green", "Blue", "Orange", "Purple"}
EMPTY_WORDS  = {"Empty", "None", "Clear", "White", "Black", "Silver", "Grey", "Gray",
                "Unclear", "Unknown", "Transparent", "Nothing"}

def ai_detect_all_slots(frame: np.ndarray, positions: Dict[int, Tuple[int, int]]) -> Dict[int, str]:
    """
    Crop each slot into a labeled grid image and send to llava-phi3 in one call.
    Each cell shows the slot number and the cropped image side by side.
    Returns a dict mapping slot number → color string.
    """
    import base64

    CELL_SIZE = 120
    COLS = 4
    ROWS = 3
    LABEL_H = 24
    CELL_H = CELL_SIZE + LABEL_H

    canvas = np.zeros((ROWS * CELL_H, COLS * CELL_SIZE, 3), dtype=np.uint8)
    slot_order = sorted(positions.keys())

    for idx, slot in enumerate(slot_order):
        row, col = divmod(idx, COLS)
        cx, cy = positions[slot]
        crop = crop_slot_zoomed(frame, cx, cy)
        if crop is None:
            crop = np.zeros((CELL_SIZE, CELL_SIZE, 3), dtype=np.uint8)
        else:
            crop = cv2.resize(crop, (CELL_SIZE, CELL_SIZE))

        y0 = row * CELL_H
        x0 = col * CELL_SIZE
        canvas[y0:y0 + LABEL_H, x0:x0 + CELL_SIZE] = (40, 40, 40)
        cv2.putText(canvas, f"Slot {slot}", (x0 + 4, y0 + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        canvas[y0 + LABEL_H:y0 + CELL_H, x0:x0 + CELL_SIZE] = crop

    cv2.imwrite(_path("annotated_scan.jpg"), canvas)
    _, buf = cv2.imencode(".jpg", canvas)
    b64 = base64.b64encode(buf.tobytes()).decode()

    slots_str = "\n".join(f"Slot {s}: [color or Empty]" for s in slot_order)
    prompt = (
        "This image shows cropped photos of lab deck slots, each labeled with its slot number. "
        "The deck base is silver/grey — ignore it. Some labware like clear plates or tip racks "
        "may also be present — treat those as Empty too. "
        "Only report a color if you clearly see a distinctly colored liquid or object: "
        "Red, Yellow, Green, Blue, Orange, or Purple. "
        "If the slot looks empty, silver, grey, clear, white, or black — reply Empty. "
        "If you are unsure, reply Empty. "
        f"Reply in exactly this format:\n{slots_str}"
    )

    response = ollama.chat(
        model="llava-phi3",
        messages=[{"role": "user", "content": prompt, "images": [b64]}],
    )
    text = response["message"]["content"]

    results = {slot: "Empty" for slot in range(1, 12)}
    for line in text.splitlines():
        line = line.strip().lstrip("*-• ").strip()
        if line.lower().startswith("slot"):
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    slot_num = int(parts[0].strip().split()[-1])
                    color_word = parts[1].strip().split()[0].capitalize().strip(".,;")
                    if 1 <= slot_num <= 11:
                        if color_word in VALID_COLORS:
                            results[slot_num] = color_word
                        else:
                            results[slot_num] = "Empty"
                except (ValueError, IndexError):
                    pass
    return results


# ── Per-slot temporal smoother ─────────────────────────────────────────────────

class SlotTracker:
    def __init__(self):
        self.history   = deque(maxlen=HISTORY_LEN)
        self.confirmed = "Empty"

    def update(self, raw_color):
        self.history.append(raw_color)
        if len(self.history) < HISTORY_LEN // 2:
            return
        counts = {}
        for c in self.history:
            counts[c] = counts.get(c, 0) + 1
        winner, votes = max(counts.items(), key=lambda x: x[1])
        if votes / len(self.history) >= CONFIRM_THRESH:
            self.confirmed = winner


# ── Main interface class ───────────────────────────────────────────────────────

class ColorDetectorInterface:
    """
    Manages the ffmpeg camera pipe and continuously updates slot trackers.
    All public methods acquire self._lock and are safe to call from any thread.
    """

    def __init__(self, video_device: str = "/dev/video2", logger=None,
                 slot_positions_file: str = "slot_positions.json"):
        self.video_device         = video_device
        self.logger               = logger
        self._slot_positions_file = slot_positions_file

        self._lock    = threading.Lock()
        self._running = False
        self._thread  = None
        self._frame   = None   # latest raw camera frame

        # Persistent data (loaded from disk)
        self._slot_positions: Dict[int, Tuple[int, int]] = {}
        self._floor_baseline: Dict[str, list]            = {}
        self._trained:        Dict[str, Tuple]           = {}

        # Per-slot trackers
        self._trackers: Dict[int, SlotTracker] = {
            sn: SlotTracker() for sn in range(1, 12)
        }

        # Training state
        self._train_color:      Optional[str] = None
        self._train_accum:      list          = []
        self._train_done_event                = threading.Event()

        self._load_all()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load_all(self):
        raw_pos = _load(self._slot_positions_file, {})
        self._slot_positions = {int(k): tuple(v) for k, v in raw_pos.items()}
        self._floor_baseline = _load("floor_baseline.json", {})
        raw_tr = _load("trained_colors.json", {})
        self._trained = {k: tuple(v) for k, v in raw_tr.items()}
        if self.logger:
            self.logger.log(
                f"Loaded: {len(self._slot_positions)} slots, "
                f"{len(self._trained)} trained colors"
            )

    def _save_slots(self):
        _save(self._slot_positions_file,
              {str(k): list(v) for k, v in self._slot_positions.items()})

    def _save_trained(self):
        _save("trained_colors.json",
              {k: list(v) for k, v in self._trained.items()})

    def _save_floor(self):
        _save("floor_baseline.json", self._floor_baseline)

    # ── Camera thread ─────────────────────────────────────────────────────────

    def start(self):
        """Start the background camera capture loop."""
        self._running = True
        self._thread  = threading.Thread(target=self._camera_loop, daemon=True)
        self._thread.start()
        if self.logger:
            self.logger.log(f"Camera loop started on {self.video_device}")

    def stop(self):
        """Stop the background camera capture loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _camera_loop(self):
        W, H = 1280, 720
        cmd = [
            "ffmpeg", "-f", "v4l2", "-input_format", "mjpeg",
            "-video_size", f"{W}x{H}", "-i", self.video_device,
            "-f", "image2pipe", "-pix_fmt", "bgr24",
            "-vcodec", "rawvideo", "-",
        ]
        pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, bufsize=10 ** 8)
        frame_count = 0
        try:
            while self._running:
                raw = pipe.stdout.read(W * H * 3)
                if len(raw) != W * H * 3:
                    break
                frame = np.frombuffer(raw, dtype="uint8").reshape((H, W, 3)).copy()
                frame_count += 1
                # Skip first 20 frames to let auto-exposure settle
                if frame_count < 20:
                    continue

                with self._lock:
                    self._frame = frame
                    self._process_frame(frame)
        finally:
            pipe.terminate()

    def _process_frame(self, frame):
        """Called inside _lock. Updates live trackers and handles training."""
        for sn, (cx, cy) in self._slot_positions.items():
            crop = crop_slot(frame, cx, cy)
            if self._trained:
                hsv        = dominant_hsv(crop)
                raw_col, _ = raw_detect(hsv, self._trained)
            else:
                raw_col = hue_range_detect(crop)
            self._trackers[sn].update(raw_col)

        # Training accumulation
        if self._train_color and len(self._train_accum) < TRAIN_FRAMES:
            sn = 1   # always train from slot 1
            if sn in self._slot_positions:
                cx, cy = self._slot_positions[sn]
                crop = crop_slot(frame, cx, cy)
                hsv  = dominant_hsv(crop)
                if hsv:
                    self._train_accum.append(hsv)
                    if len(self._train_accum) >= TRAIN_FRAMES:
                        arr = np.array(self._train_accum)
                        med = tuple(np.median(arr, axis=0).tolist())
                        self._trained[self._train_color] = med
                        self._save_trained()
                        if self.logger:
                            h, s, v = med
                            self.logger.log(
                                f"Trained {self._train_color}: "
                                f"H={h:.1f} S={s:.1f} V={v:.1f}"
                            )
                        self._train_color = None
                        self._train_accum = []
                        self._train_done_event.set()

    # ── Public API (thread-safe) ──────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running and (self._thread is not None) and self._thread.is_alive()

    def get_all_colors(self) -> Dict[int, str]:
        """Return the live tracker-confirmed color for every slot."""
        with self._lock:
            return {sn: tr.confirmed for sn, tr in self._trackers.items()}

    def get_trained_colors(self) -> list:
        with self._lock:
            return list(self._trained.keys())

    def get_slots_mapped(self) -> int:
        with self._lock:
            return len(self._slot_positions)

    def has_floor_baseline(self) -> bool:
        with self._lock:
            return bool(self._floor_baseline)

    def set_slot_position(self, slot: int, x: int, y: int):
        with self._lock:
            self._slot_positions[slot] = (x, y)
            self._save_slots()

    def record_floor_baseline(self) -> Dict[str, list]:
        """Sample all mapped slots with an empty deck and save as the floor baseline."""
        with self._lock:
            frame     = self._frame
            positions = dict(self._slot_positions)
        if frame is None:
            raise RuntimeError("No camera frame available yet.")
        baseline = {}
        for sn, (cx, cy) in positions.items():
            hsv = mean_hsv(crop_slot(frame, cx, cy))
            if hsv:
                baseline[str(sn)] = list(hsv)
        with self._lock:
            self._floor_baseline = baseline
            self._save_floor()
        return baseline

    def start_training(self, color_name: str) -> threading.Event:
        """
        Begin training for color_name. Place the cube in Slot 1 first.
        Returns a threading.Event that is set when training completes.
        """
        with self._lock:
            self._train_color      = color_name
            self._train_accum      = []
            self._train_done_event = threading.Event()
            evt = self._train_done_event
        return evt

    def capture_snapshot(self, path: str) -> bool:
        """Save the current frame to disk. Returns True on success."""
        with self._lock:
            frame = self._frame
        if frame is None:
            return False
        cv2.imwrite(path, frame)
        return True

    def scan_environment_zoomed(self) -> Dict[int, str]:
        """
        One-shot zoomed scan of all 11 slots.

        For each slot:
          1. Crop a ZOOM_CROP_SIZE region around the slot center.
          2. Upscale to ZOOM_OUT_SIZE x ZOOM_OUT_SIZE for more pixel detail.
          3. If trained colors exist → nearest-centroid HSV detection.
             Otherwise → hue-range detection (Red/Yellow/Green/Blue, no training needed).

        Returns a dict mapping slot number → detected color string.
        """
        with self._lock:
            frame     = self._frame
            positions = dict(self._slot_positions)
            trained   = dict(self._trained)

        if frame is None:
            raise RuntimeError("No camera frame available yet.")

        results = {}
        for slot in range(1, 12):
            if slot not in positions:
                results[slot] = "Unmapped"
                continue
            cx, cy = positions[slot]
            crop = crop_slot_zoomed(frame, cx, cy)
            results[slot] = hue_range_detect(crop)

        # Save annotated grid for debugging
        CELL_SIZE = 120
        COLS = 4
        ROWS = 3
        LABEL_H = 24
        CELL_H = CELL_SIZE + LABEL_H
        canvas = np.zeros((ROWS * CELL_H, COLS * CELL_SIZE, 3), dtype=np.uint8)
        for idx, slot in enumerate(sorted(positions.keys())):
            row, col = divmod(idx, COLS)
            cx, cy = positions[slot]
            crop = crop_slot_zoomed(frame, cx, cy)
            cell = np.zeros((CELL_SIZE, CELL_SIZE, 3), dtype=np.uint8) if crop is None \
                else cv2.resize(crop, (CELL_SIZE, CELL_SIZE))
            y0, x0 = row * CELL_H, col * CELL_SIZE
            canvas[y0:y0 + LABEL_H, x0:x0 + CELL_SIZE] = (40, 40, 40)
            label = f"S{slot}: {results[slot]}"
            cv2.putText(canvas, label, (x0 + 2, y0 + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            canvas[y0 + LABEL_H:y0 + CELL_H, x0:x0 + CELL_SIZE] = cell
        cv2.imwrite(_path("annotated_scan.jpg"), canvas)

        return results
