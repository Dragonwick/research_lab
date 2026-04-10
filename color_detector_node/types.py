"""Pydantic types shared between the interface and node."""

from typing import Dict
from pydantic import BaseModel, Field


class EnvironmentScan(BaseModel):
    """Full color scan result for all 11 OT-2 deck slots."""
    slots: Dict[int, str] = Field(
        ...,
        description="Slot number → detected color (or 'Empty' / 'Unclear' / 'Unmapped')",
    )
    empty_count:    int = Field(..., description="Slots confirmed empty")
    detected_count: int = Field(..., description="Slots with a confirmed color")
    unclear_count:  int = Field(..., description="Slots whose color could not be determined")


class TrainColorRequest(BaseModel):
    """Request to train a specific color."""
    color_name: str = Field(
        ...,
        description="Color to train: Blue, Red, Orange, Yellow, Purple, or Green",
    )


class SetSlotPositionRequest(BaseModel):
    """Set the pixel coordinates for a single slot on the camera frame."""
    slot: int = Field(..., description="Slot number (1-11)")
    x:    int = Field(..., description="Pixel x-coordinate of slot center")
    y:    int = Field(..., description="Pixel y-coordinate of slot center")
