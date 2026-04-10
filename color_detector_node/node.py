"""
OT-2 Color Detector — MADSci RestNode
--------------------------------------
Exposes the color detector as a REST service.

Actions:
  POST /action/scan_environment      → EnvironmentScan  (main action)
  POST /action/train_color           → {"status": "ok", "color": ...}
  POST /action/record_floor_baseline → {"status": "ok", "slots": ...}
  POST /action/capture_snapshot      → {"status": "ok", "path": ...}

GET  /state  → live node status (auto-updated every 2 s)
"""

from typing import Any, Dict
from pydantic import Field

from madsci.node_module import RestNode
from madsci.node_module.helpers import action
from madsci.common.types.node_types import RestNodeConfig

from .interface import ColorDetectorInterface
from .types import (
    EnvironmentScan,
    TrainColorRequest,
    SetSlotPositionRequest,
)

import os

HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(HERE)


# ── Config ─────────────────────────────────────────────────────────────────────

class ColorDetectorConfig(RestNodeConfig):
    """Configuration for the Color Detector node."""
    video_device: str = Field(
        default="/dev/video0",
        description="V4L2 device path for the primary OT-2 camera",
    )
    video_device_2: str = Field(
        default="/dev/video0",
        description="V4L2 device path for the secondary camera",
    )
    snapshot_dir: str = Field(
        default=DATA_DIR,
        description="Directory to save snapshot images",
    )


# ── Node ───────────────────────────────────────────────────────────────────────

class ColorDetectorNode(RestNode):
    """MADSci node that detects colors of labware cubes in OT-2 deck slots."""

    config:       ColorDetectorConfig = ColorDetectorConfig()
    config_model  = ColorDetectorConfig

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def startup_handler(self) -> None:
        self.logger.log("Starting Color Detector node...")
        self.camera = ColorDetectorInterface(
            video_device=self.config.video_device,
            logger=self.logger,
            slot_positions_file="slot_positions.json",
        )
        self.camera.start()
        if self.config.video_device_2 != self.config.video_device:
            self.camera2 = ColorDetectorInterface(
                video_device=self.config.video_device_2,
                logger=self.logger,
                slot_positions_file="logitech_slot_positions.json",
            )
            self.camera2.start()
            self.logger.log("Both cameras started.")
        else:
            self.camera2 = None
            self.logger.log("Single camera mode (camera2 skipped — same device).")

    def shutdown_handler(self) -> None:
        self.logger.log("Shutting down Color Detector node.")
        if hasattr(self, "camera"):
            self.camera.stop()
        if hasattr(self, "camera2") and self.camera2:
            self.camera2.stop()

    def _merge(self, colors1: Dict, colors2: Dict) -> Dict:
        """
        Merge results from two cameras.
        Agreement → use that color.
        One says Unclear/Unmapped → use the other.
        Disagreement → mark Unclear.
        """
        merged = {}
        for slot in range(1, 12):
            c1 = colors1.get(slot, "Unclear")
            c2 = colors2.get(slot, "Unclear")
            uncertain = {"Unclear", "Unmapped"}
            if c1 == c2:
                merged[slot] = c1
            elif c1 in uncertain:
                merged[slot] = c2
            elif c2 in uncertain:
                merged[slot] = c1
            else:
                merged[slot] = "Unclear"
        return merged

    def state_handler(self) -> Dict[str, Any]:
        """Called every ~2 s by MADSci to keep the node state current."""
        if not hasattr(self, "camera"):
            return {}
        c2_colors = self.camera2.get_all_colors() if self.camera2 else {}
        colors = self._merge(self.camera.get_all_colors(), c2_colors)
        self.node_state = {
            "camera1_running":    self.camera.is_running,
            "camera2_running":    self.camera2.is_running if self.camera2 else False,
            "slots_mapped":       self.camera.get_slots_mapped(),
            "floor_baseline_set": self.camera.has_floor_baseline(),
            "trained_colors":     self.camera.get_trained_colors(),
            "current_colors":     {str(k): v for k, v in colors.items()},
        }
        return self.node_state

    # ── Actions ───────────────────────────────────────────────────────────────

    @action
    def scan_environment(self) -> EnvironmentScan:
        """
        Zoom into each of the 11 deck slots on both cameras and return the
        detected color for every slot.

        Each slot is cropped at a larger radius than the live tracker uses,
        then upscaled before running HSV color detection — giving sharper,
        more reliable reads than the continuous background scan.

        Both cameras are consulted; if they agree the result is used directly.
        If they disagree, the slot is marked 'Unclear'.
        """
        c2_colors = self.camera2.scan_environment_zoomed() if self.camera2 else {}
        colors = self._merge(self.camera.scan_environment_zoomed(), c2_colors)
        empty    = sum(1 for c in colors.values() if c == "Empty")
        detected = sum(1 for c in colors.values() if c not in ("Empty", "Unclear", "Unmapped"))
        unclear  = sum(1 for c in colors.values() if c in ("Unclear", "Unmapped"))
        return EnvironmentScan(
            slots=colors,
            empty_count=empty,
            detected_count=detected,
            unclear_count=unclear,
        )

    @action
    def train_color(self, request: TrainColorRequest) -> dict:
        """
        Train a color signature. Place the cube in Slot 1 FIRST, then call
        this action. Blocks until sampling completes (~2 s).
        """
        valid = ["Blue", "Red", "Orange", "Yellow", "Purple", "Green"]
        if request.color_name not in valid:
            raise ValueError(
                f"Unknown color '{request.color_name}'. Must be one of {valid}"
            )
        self.logger.log(f"Training {request.color_name} — sampling Slot 1...")
        done_event = self.camera.start_training(request.color_name)
        done_event.wait(timeout=15)
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
        Sample all slot positions with an EMPTY deck and save as the floor
        baseline. Call this once before scanning for best accuracy.
        """
        self.logger.log("Recording floor baseline — deck must be empty.")
        baseline = self.camera.record_floor_baseline()
        return {
            "status": "ok",
            "slots_sampled": len(baseline),
            "message": "Floor baseline saved. You can now place cubes and scan.",
        }

    @action
    def capture_snapshot(self, filename: str = "snapshot.jpg") -> dict:
        """Save the current camera frame to the snapshot directory."""
        path = os.path.join(self.config.snapshot_dir, filename)
        ok   = self.camera.capture_snapshot(path)
        if not ok:
            raise RuntimeError("No camera frame available — is the camera running?")
        return {"status": "ok", "path": path}

    @action
    def set_slot_position(self, request: SetSlotPositionRequest) -> dict:
        """Update the pixel coordinates for one slot on the camera frame."""
        self.camera.set_slot_position(request.slot, request.x, request.y)
        return {
            "status": "ok",
            "slot": request.slot,
            "position": {"x": request.x, "y": request.y},
        }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    node = ColorDetectorNode()
    node.start_node()
