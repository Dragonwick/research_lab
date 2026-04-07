"""Pydantic types shared between the interface and node."""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class SlotColor(BaseModel):
    """Color result for a single OT-2 deck slot."""
    slot: int = Field(..., description="Slot number (1-11)")
    color: str = Field(..., description="Detected color name or 'Empty'/'Unclear'")
    confirmed: bool = Field(..., description="True if color passed the stability threshold")


class TrayColorMap(BaseModel):
    """Color results for all 11 OT-2 deck slots."""
    slots: Dict[int, str] = Field(
        ...,
        description="Mapping of slot number → color name for all 11 slots",
    )
    empty_count: int = Field(..., description="Number of slots currently empty")
    detected_count: int = Field(..., description="Number of slots with a confirmed color")


class TrainColorRequest(BaseModel):
    """Request to train a specific color."""
    color_name: str = Field(
        ...,
        description="Color to train: Blue, Red, Orange, Yellow, Purple, or Green",
    )
    frames: int = Field(40, description="Number of frames to average (default 40)")


class SetSlotPositionRequest(BaseModel):
    """Set the pixel coordinates for a single slot on the camera frame."""
    slot: int = Field(..., description="Slot number (1-11)")
    x: int = Field(..., description="Pixel x-coordinate of slot center")
    y: int = Field(..., description="Pixel y-coordinate of slot center")


class NodeStatus(BaseModel):
    """Live status of the color detector node."""
    camera_running: bool
    slots_mapped: int
    floor_baseline_set: bool
    trained_colors: list[str]
    current_colors: Dict[int, str]
