"""
Color Mixing Experiment — MADSci Edition
-----------------------------------------
Uses RestNodeClient to talk to both MADSci nodes:
  • color_detector_node  (http://localhost:2000) — scans the deck
  • ot2_node             (http://localhost:2001) — runs the pipette

Usage:
    python mix.py
"""

import json
from madsci.client import RestNodeClient
from madsci.common.types.action_types import ActionRequest

COLOR_NODE_URL = "http://localhost:2000"
OT2_NODE_URL   = "http://localhost:2001"
TIPRACK_SLOT   = 1     # slot that holds tips — excluded from dest candidates
TRANSFER_VOL   = 100   # µL per color


def scan_deck() -> dict:
    """Call the color detector node and return slot→color map."""
    print("  Connecting to color detector node...")
    client = RestNodeClient(url=COLOR_NODE_URL)
    result = client.send_action(
        ActionRequest(action_name="scan_environment"),
        await_result=True,
        timeout=120,
    )
    if result.status != "succeeded":
        raise RuntimeError(f"Scan failed: {result.errors}")
    return result.json_result["slots"]


def find_slots(slots: dict) -> tuple:
    """Find red slot, blue slot, and an empty destination slot from scan results."""
    red_slot  = next((int(k) for k, v in slots.items() if v == "Red"),  None)
    blue_slot = next((int(k) for k, v in slots.items() if v == "Blue"), None)
    dest_slot = next(
        (int(k) for k, v in slots.items()
         if v == "Empty" and int(k) != TIPRACK_SLOT),
        None,
    )
    if red_slot is None:
        raise RuntimeError("No Red well detected. Check camera and slot positions.")
    if blue_slot is None:
        raise RuntimeError("No Blue well detected. Check camera and slot positions.")
    if dest_slot is None:
        raise RuntimeError("No empty destination slot available.")
    return red_slot, blue_slot, dest_slot


def run_mix(red_slot: int, blue_slot: int, dest_slot: int, volume: int) -> dict:
    """Tell the OT-2 node to execute the mixing protocol."""
    print("  Connecting to OT-2 node...")
    client = RestNodeClient(url=OT2_NODE_URL)
    result = client.send_action(
        ActionRequest(
            action_name="run_mix",
            args={
                "red_slot":  red_slot,
                "blue_slot": blue_slot,
                "dest_slot": dest_slot,
                "volume":    volume,
            },
        ),
        await_result=True,
        timeout=600,
    )
    if result.status != "succeeded":
        raise RuntimeError(f"Mix failed: {result.errors}")
    return result.json_result


def main():
    print("=== Color Mixing Experiment (MADSci) ===\n")

    print("Step 1: Scanning deck...")
    slots = scan_deck()
    print("  Detected colors:")
    for slot, color in sorted(slots.items(), key=lambda x: int(x[0])):
        print(f"    Slot {slot}: {color}")

    red_slot, blue_slot, dest_slot = find_slots(slots)
    print(f"\n  Plan → Red: slot {red_slot} | Blue: slot {blue_slot} | Dest: slot {dest_slot}")

    confirm = input("\nProceed with mixing? (y/n): ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    print("\nStep 2: Running mix on OT-2...")
    result = run_mix(red_slot, blue_slot, dest_slot, TRANSFER_VOL)
    print(f"  {result.get('message', 'Done.')}")
    print(f"  Run ID: {result.get('run_id')}")

    print("\nDone! Check the deck for the mixed result.")


if __name__ == "__main__":
    main()
