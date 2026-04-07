"""
OT-2 Color Detector — MADSci RestNode
--------------------------------------
Exposes the color detector as a REST service that can be orchestrated
by a MADSci Workcell Manager alongside the OT-2 robot node.

Endpoints (via @action):
  POST /action/scan_all_slots       → TrayColorMap
  POST /action/get_slot_color       → SlotColor
  POST /action/get_all_colors       → TrayColorMap
  POST /action/train_color          → {"status": "ok", "color": ...}
  POST /action/record_floor_baseline → {"status": "ok", "slots": ...}
  POST /action/set_slot_position    → {"status": "ok"}
  POST /action/capture_snapshot     → {"status": "ok", "path": ...}

GET  /state   → NodeStatus (auto-updated every 2 s)
"""

from typing import Any, Dict
from pydantic import Field

from madsci.node_module import RestNode
from madsci.node_module.helpers import action
from madsci.common.types.node_types import RestNodeConfig

from .interface import ColorDetectorInterface
from .types import (
    SlotColor,
    TrayColorMap,
    TrainColorRequest,
    SetSlotPositionRequest,
    NodeStatus,
)

import os

HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(HERE)


# ── Config ─────────────────────────────────────────────────────────────────────

class ColorDetectorConfig(RestNodeConfig):
    """Configuration for the Color Detector node.

    All fields can be overridden via environment variables with the
    NODE_ prefix (e.g. NODE_VIDEO_DEVICE=/dev/video0).
    """
    video_device: str = Field(
        default="/dev/video2",
        description="V4L2 device path for the OT-2 camera",
    )
    snapshot_dir: str = Field(
        default=DATA_DIR,
        description="Directory to save snapshot images",
    )


# ── Node ───────────────────────────────────────────────────────────────────────

class ColorDetectorNode(RestNode):
    """MADSci node that detects colors of labware cubes in OT-2 deck slots."""

    config: ColorDetectorConfig = ColorDetectorConfig()
    config_model = ColorDetectorConfig

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def startup_handler(self) -> None:
        self.logger.log("Starting Color Detector node...")
        self.camera = ColorDetectorInterface(
            video_device=self.config.video_device,
            logger=self.logger,
        )
        self.camera.start()
        self.logger.log("Camera interface started.")

    def shutdown_handler(self) -> None:
        self.logger.log("Shutting down Color Detector node.")
        if hasattr(self, "camera"):
            self.camera.stop()

    def state_handler(self) -> Dict[str, Any]:
        """Called every ~2 s by MADSci to keep the node state current."""
        if not hasattr(self, "camera"):
            return {}
        colors = self.camera.get_all_colors()
        self.node_state = {
            "camera_running":    self.camera.is_running,
            "slots_mapped":      self.camera.get_slots_mapped(),
            "floor_baseline_set": self.camera.has_floor_baseline(),
            "trained_colors":    self.camera.get_trained_colors(),
            "current_colors":    {str(k): v for k, v in colors.items()},
        }
        return self.node_state

    # ── Actions ───────────────────────────────────────────────────────────────

    @action
    def scan_all_slots(self, frames_per_slot: int = 20) -> TrayColorMap:
        """
        Wait for frames_per_slot frames of data then return the confirmed
        color for every OT-2 deck slot (1-11).
        """
        colors = self.camera.scan_all_slots(frames_per_slot=frames_per_slot)
        empty   = sum(1 for c in colors.values() if c == "Empty")
        detected = sum(1 for c in colors.values() if c not in ("Empty", "Unclear"))
        return TrayColorMap(
            slots=colors,
            empty_count=empty,
            detected_count=detected,
        )

    @action
    def get_slot_color(self, slot: int) -> SlotColor:
        """Return the current confirmed color for a single slot."""
        color = self.camera.get_slot_color(slot)
        return SlotColor(
            slot=slot,
            color=color,
            confirmed=color not in ("Unclear",),
        )

    @action
    def get_all_colors(self) -> TrayColorMap:
        """Return live confirmed colors for all 11 slots without waiting."""
        colors  = self.camera.get_all_colors()
        empty   = sum(1 for c in colors.values() if c == "Empty")
        detected = sum(1 for c in colors.values() if c not in ("Empty", "Unclear"))
        return TrayColorMap(
            slots=colors,
            empty_count=empty,
            detected_count=detected,
        )

    @action
    def train_color(self, request: TrainColorRequest) -> dict:
        """
        Train a color signature. Place the target cube in Slot 1 FIRST,
        then call this action. Blocks until training completes (~2 s).
        """
        valid = ["Blue", "Red", "Orange", "Yellow", "Purple", "Green"]
        if request.color_name not in valid:
            raise ValueError(
                f"Unknown color '{request.color_name}'. Must be one of {valid}"
            )
        self.logger.log(f"Training {request.color_name} — sampling Slot 1...")
        done_event = self.camera.start_training(request.color_name)
        done_event.wait(timeout=15)   # wait up to 15 s
        if not done_event.is_set():
            raise RuntimeError(
                "Training timed out. Make sure a cube is visible in Slot 1."
            )
        return {
            "status": "ok",
            "color": request.color_name,
            "trained_colors": self.camera.get_trained_colors(),
        }

    @action
    def record_floor_baseline(self) -> dict:
        """
        Sample all slot positions with an EMPTY deck and save as the
        floor baseline. Call this before scanning for best accuracy.
        """
        self.logger.log("Recording floor baseline — deck must be empty.")
        baseline = self.camera.record_floor_baseline()
        return {
            "status": "ok",
            "slots_sampled": len(baseline),
            "message": "Floor baseline saved. You can now place cubes and scan.",
        }

    @action
    def set_slot_position(self, request: SetSlotPositionRequest) -> dict:
        """Update the pixel coordinates for one slot on the camera frame."""
        self.camera.set_slot_position(request.slot, request.x, request.y)
        return {
            "status": "ok",
            "slot": request.slot,
            "position": {"x": request.x, "y": request.y},
        }

    @action
    def capture_snapshot(self, filename: str = "snapshot.jpg") -> dict:
        """Save the current camera frame to the snapshot directory."""
        import os
        path = os.path.join(self.config.snapshot_dir, filename)
        ok   = self.camera.capture_snapshot(path)
        if not ok:
            raise RuntimeError("No camera frame available — is the camera running?")
        return {"status": "ok", "path": path}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    node = ColorDetectorNode()
    node.start_node()
