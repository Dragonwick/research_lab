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


# ── Detection constants ────────────────────────────────────────────────────────
SLOT_CROP_SIZE = 80
TRAIN_FRAMES   = 40
HISTORY_LEN    = 30
CONFIRM_THRESH = 0.55
MAX_COLOR_DIST = 6000
SAT_CUT        = 50      # min saturation for colorful-pixel filter

HERE           = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(os.path.dirname(HERE))   # ot2files/


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

def crop_slot(frame, cx, cy):
    h, w = frame.shape[:2]
    half = SLOT_CROP_SIZE // 2
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(w, cx + half), min(h, cy + half)
    r = frame[y1:y2, x1:x2]
    return r if r.size > 0 else None


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
        self.video_device       = video_device
        self.logger             = logger
        self._slot_positions_file = slot_positions_file

        self._lock     = threading.Lock()
        self._running  = False
        self._thread   = None
        self._frame    = None   # latest raw camera frame

        # Persistent data (loaded from disk)
        self._slot_positions: Dict[int, Tuple[int, int]] = {}
        self._floor_baseline: Dict[str, list]            = {}
        self._trained:        Dict[str, Tuple]           = {}

        # Per-slot trackers
        self._trackers: Dict[int, SlotTracker] = {
            sn: SlotTracker() for sn in range(1, 12)
        }

        # Training state
        self._train_color:  Optional[str]  = None
        self._train_accum:  list           = []
        self._train_done_event             = threading.Event()

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
        cmd = [
            "ffmpeg", "-f", "v4l2", "-video_size", "640x480",
            "-i", self.video_device,
            "-f", "image2pipe", "-pix_fmt", "bgr24",
            "-vcodec", "rawvideo", "-",
        ]
        pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, bufsize=10 ** 8)
        try:
            while self._running:
                raw = pipe.stdout.read(640 * 480 * 3)
                if len(raw) != 640 * 480 * 3:
                    break
                frame = np.frombuffer(raw, dtype="uint8").reshape((480, 640, 3)).copy()

                with self._lock:
                    self._frame = frame
                    self._process_frame(frame)
        finally:
            pipe.terminate()

    def _process_frame(self, frame):
        """Called inside _lock. Updates trackers and handles training."""
        for sn, (cx, cy) in self._slot_positions.items():
            crop    = crop_slot(frame, cx, cy)
            hsv     = dominant_hsv(crop)
            raw_col, _ = raw_detect(hsv, self._trained)
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
                        arr  = np.array(self._train_accum)
                        med  = tuple(np.median(arr, axis=0).tolist())
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

    def get_slot_color(self, slot: int) -> str:
        with self._lock:
            tr = self._trackers.get(slot)
            return tr.confirmed if tr else "Unknown"

    def get_all_colors(self) -> Dict[int, str]:
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

    def set_all_slot_positions(self, positions: Dict[int, Tuple[int, int]]):
        with self._lock:
            self._slot_positions = dict(positions)
            self._save_slots()

    def record_floor_baseline(self) -> Dict[str, list]:
        """Sample all currently mapped slots and save as the floor baseline."""
        with self._lock:
            frame = self._frame
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
        The caller can wait on it or poll it.
        """
        with self._lock:
            self._train_color = color_name
            self._train_accum = []
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

    def scan_all_slots(self, frames_per_slot: int = 20) -> Dict[int, str]:
        """
        Block for frames_per_slot frames per slot and return the majority vote
        for each of the 11 slots. Uses the live trackers — no extra waiting
        needed if trackers are already warm.
        """
        import time
        # Give trackers time to warm up if they haven't yet
        if not self.is_running:
            raise RuntimeError("Camera is not running.")
        time.sleep(frames_per_slot / 30.0)   # ~30 fps → wait that many frames
        return self.get_all_colors()
