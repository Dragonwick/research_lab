"""
OT-2 Robot — MADSci RestNode
-----------------------------
Wraps the Opentrons OT-2 HTTP API as a MADSci REST node.

Actions:
  POST /action/run_mix          → upload and run a two-color mixing protocol
  POST /action/get_robot_status → returns current robot status
  POST /action/stop_run         → stop any running protocol
"""

import time
from typing import Any, Dict, Optional

import requests
from pydantic import Field

from madsci.node_module import RestNode
from madsci.node_module.helpers import action
from madsci.common.types.node_types import RestNodeConfig


# ── Config ─────────────────────────────────────────────────────────────────────

class OT2Config(RestNodeConfig):
    """Configuration for the OT-2 node."""
    node_url: str = Field(
        default="http://127.0.0.1:2001/",
        description="URL this node listens on",
    )
    ot2_url: str = Field(
        default="http://169.254.19.251:31950",
        description="Base URL of the OT-2 HTTP API",
    )
    pipette_mount: str = Field(
        default="right",
        description="Mount position of the pipette (left or right)",
    )
    tiprack_slot: int = Field(
        default=1,
        description="Deck slot holding the tip rack",
    )
    tiprack_name: str = Field(
        default="opentrons_96_tiprack_300ul",
        description="Labware name of the tip rack",
    )
    source_labware: str = Field(
        default="nest_1_reservoir_195ml",
        description="Labware name of the source reservoirs (red/blue wells)",
    )
    dest_labware: str = Field(
        default="corning_96_wellplate_360ul_flat",
        description="Labware name of the destination plate",
    )
    run_timeout: int = Field(
        default=300,
        description="Max seconds to wait for a protocol run to complete",
    )


# ── Node ───────────────────────────────────────────────────────────────────────

class OT2Node(RestNode):
    """MADSci node for the Opentrons OT-2 liquid handling robot."""

    config:       OT2Config = OT2Config()
    config_model  = OT2Config

    @property
    def _headers(self) -> Dict[str, str]:
        return {"Opentrons-Version": "3"}

    @property
    def _json_headers(self) -> Dict[str, str]:
        return {**self._headers, "Content-Type": "application/json"}

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def startup_handler(self) -> None:
        self.logger.log(f"OT-2 node starting — robot at {self.config.ot2_url}")
        self._current_run_id: Optional[str] = None

    def state_handler(self) -> Dict[str, Any]:
        try:
            r = requests.get(f"{self.config.ot2_url}/health",
                             headers=self._headers, timeout=3)
            health = r.json()
            return {
                "robot_name":    health.get("name", "unknown"),
                "api_version":   health.get("api_version", "unknown"),
                "current_run_id": self._current_run_id,
            }
        except Exception:
            return {"reachable": False, "current_run_id": self._current_run_id}

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _build_protocol(self, red_slot: int, blue_slot: int,
                        dest_slot: int, volume: int) -> str:
        cfg = self.config
        return f"""
from opentrons import protocol_api

metadata = {{"apiLevel": "2.15"}}

def run(protocol: protocol_api.ProtocolContext):
    tiprack  = protocol.load_labware("{cfg.tiprack_name}", {cfg.tiprack_slot})
    pipette  = protocol.load_instrument("p300_single_gen2", "{cfg.pipette_mount}", tip_racks=[tiprack])

    red_well  = protocol.load_labware("{cfg.source_labware}", {red_slot}).wells()[0]
    blue_well = protocol.load_labware("{cfg.source_labware}", {blue_slot}).wells()[0]
    dest_well = protocol.load_labware("{cfg.dest_labware}", {dest_slot}).wells()[0]

    pipette.pick_up_tip()
    pipette.aspirate({volume}, red_well)
    pipette.dispense({volume}, dest_well)
    pipette.drop_tip()

    pipette.pick_up_tip()
    pipette.aspirate({volume}, blue_well)
    pipette.dispense({volume}, dest_well)
    pipette.drop_tip()

    protocol.comment("Mix complete — check slot {dest_slot} for purple.")
"""

    def _upload_and_run(self, protocol_text: str) -> str:
        """Upload a protocol and start a run. Returns the run_id."""
        url = self.config.ot2_url

        # Upload
        files = {"files": ("protocol.py", protocol_text.encode(), "text/x-python")}
        r = requests.post(f"{url}/protocols", headers=self._headers, files=files)
        r.raise_for_status()
        protocol_id = r.json()["data"]["id"]
        self.logger.log(f"Protocol uploaded: {protocol_id}")

        # Create run
        r = requests.post(f"{url}/runs", headers=self._json_headers,
                          json={"data": {"protocolId": protocol_id}})
        r.raise_for_status()
        run_id = r.json()["data"]["id"]
        self._current_run_id = run_id
        self.logger.log(f"Run created: {run_id}")

        # Start run
        r = requests.post(f"{url}/runs/{run_id}/actions",
                          headers=self._json_headers,
                          json={"data": {"actionType": "play"}})
        r.raise_for_status()
        self.logger.log(f"Run started: {run_id}")

        # Poll until done
        deadline = time.time() + self.config.run_timeout
        while time.time() < deadline:
            time.sleep(3)
            status = requests.get(f"{url}/runs/{run_id}",
                                  headers=self._headers).json()["data"]["status"]
            self.logger.log(f"Run status: {status}")
            if status == "succeeded":
                self._current_run_id = None
                return run_id
            if status in ("failed", "stopped"):
                self._current_run_id = None
                raise RuntimeError(f"OT-2 run ended with status: {status}")

        raise TimeoutError(f"OT-2 run timed out after {self.config.run_timeout}s")

    # ── Actions ────────────────────────────────────────────────────────────────

    @action
    def run_mix(self, red_slot: int, blue_slot: int,
                dest_slot: int, volume: int = 100) -> dict:
        """
        Build and run a two-color mixing protocol.
        Transfers `volume` µL from red_slot and blue_slot into dest_slot.
        """
        self.logger.log(
            f"Mixing: red={red_slot} + blue={blue_slot} → dest={dest_slot} ({volume}µL each)"
        )
        protocol = self._build_protocol(red_slot, blue_slot, dest_slot, volume)
        run_id = self._upload_and_run(protocol)
        return {
            "status": "ok",
            "run_id": run_id,
            "red_slot": red_slot,
            "blue_slot": blue_slot,
            "dest_slot": dest_slot,
            "volume_ul": volume,
            "message": f"Mixed {volume}µL red + {volume}µL blue into slot {dest_slot}.",
        }

    @action
    def get_robot_status(self) -> dict:
        """Return the current OT-2 health and run status."""
        try:
            r = requests.get(f"{self.config.ot2_url}/health",
                             headers=self._headers, timeout=5)
            r.raise_for_status()
            health = r.json()
            runs_r = requests.get(f"{self.config.ot2_url}/runs",
                                  headers=self._headers, timeout=5)
            runs = runs_r.json().get("data", [])
            active = next((x for x in runs if x.get("status") == "running"), None)
            return {
                "reachable":   True,
                "name":        health.get("name"),
                "api_version": health.get("api_version"),
                "active_run":  active.get("id") if active else None,
            }
        except Exception as e:
            return {"reachable": False, "error": str(e)}

    @action
    def stop_run(self) -> dict:
        """Stop any currently running protocol on the OT-2."""
        run_id = self._current_run_id
        if not run_id:
            # Try to find a running run
            runs = requests.get(f"{self.config.ot2_url}/runs",
                                headers=self._headers, timeout=5).json().get("data", [])
            active = next((x for x in runs if x.get("status") == "running"), None)
            if not active:
                return {"status": "ok", "message": "No active run to stop."}
            run_id = active["id"]

        requests.post(f"{self.config.ot2_url}/runs/{run_id}/actions",
                      headers=self._json_headers,
                      json={"data": {"actionType": "stop"}})
        self._current_run_id = None
        return {"status": "ok", "run_id": run_id, "message": "Run stopped."}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    node = OT2Node()
    node.start_node()
