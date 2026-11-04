"""
CrowdPulse - Dispatch & Response Services (Phase 4)
Project: CrowdPulse - Crowd Density Management System (SIH25165)
Phase: Phase 4 Multi-Agent Coordination

Beginner-Friendly Explanation:
This module connects the Dispatch Agent's decisions to physical field equipment:
1. MarshalSquadTracker: Tracks on-ground security teams (Squad Alpha, Ushers, Medical)
2. TurnstileGateController: Controls automated turnstiles (Throttling, Lock, Evacuation)
3. DigitalSignageService: Updates overhead LED directional wayfinding screens
4. AlertNotificationService: Formulates simulated SMS / Email / Webhook alerts
"""

from datetime import datetime
from typing import Dict, List, Optional
import config
from agents.schemas import utc_now


class MarshalSquadTracker:
    """Tracks field security squads, their locations, and availability status."""

    def __init__(self):
        self.squads = {
            "SQUAD-ALPHA": {
                "name": "Rapid Response Squad Alpha",
                "officers": 6,
                "current_zone": "Central Plaza",
                "status": "AVAILABLE",  # "AVAILABLE", "DISPATCHED", "ON_SCENE"
                "equipment": ["Crowd Megaphones", "Barriers", "First Aid"],
            },
            "SQUAD-BRAVO": {
                "name": "Crowd Flow Ushers Team Bravo",
                "officers": 4,
                "current_zone": "North Gate",
                "status": "AVAILABLE",
                "equipment": ["Directorial Signage", "Megaphones"],
            },
            "MED-UNIT-1": {
                "name": "Emergency Medical Response 01",
                "officers": 3,
                "current_zone": "South Exit Hub",
                "status": "AVAILABLE",
                "equipment": ["Mobile Defibrillator", "Stretcher", "First Aid"],
            },
        }

    def dispatch_squad(self, squad_id: str, target_zone_name: str, reason: str) -> Dict:
        """Dispatches a squad to an incident zone."""
        if squad_id in self.squads:
            squad = self.squads[squad_id]
            squad["status"] = "DISPATCHED"
            squad["current_zone"] = target_zone_name
            squad["dispatch_time"] = utc_now().strftime("%H:%M:%S")
            squad["mission"] = reason
            return squad
        return {}

    def get_all_squads(self) -> List[Dict]:
        return [{"id": k, **v} for k, v in self.squads.items()]


class TurnstileGateController:
    """Simulates automated electronic turnstiles and perimeter gates."""

    def __init__(self):
        # Default state for all 4 zones
        self.gates = {
            z["id"]: {
                "zone_name": z["name"],
                "gate_status": "NORMAL",  # "NORMAL", "METERED", "LOCKED_INFLOW", "EVACUATION_OPEN"
                "flow_rate_limit_ppm": 60,  # Persons per minute
            }
            for z in config.DEFAULT_ZONES
        }

    def set_gate_mode(self, zone_id: str, mode: str, rate_limit_ppm: int = 60) -> Dict:
        """Updates gate operating mode."""
        if zone_id in self.gates:
            self.gates[zone_id]["gate_status"] = mode
            self.gates[zone_id]["flow_rate_limit_ppm"] = rate_limit_ppm
            return self.gates[zone_id]
        return {}

    def get_all_gates(self) -> List[Dict]:
        return [{"zone_id": k, **v} for k, v in self.gates.items()]


class DigitalSignageService:
    """Simulates dynamic overhead LED wayfinding signage throughout the venue."""

    def __init__(self):
        self.active_messages = [
            {"screen_id": "SCREEN-N01", "location": "North Entrance", "text": "Welcome to CrowdPulse Arena | Gates 1-4 Open", "color": "GREEN"},
            {"screen_id": "SCREEN-C01", "location": "Central Concourse", "text": "Keep Moving Towards Central Plaza", "color": "GREEN"},
            {"screen_id": "SCREEN-S01", "location": "South Exit / Transit", "text": "All Transit Escalators Operational", "color": "GREEN"},
        ]

    def broadcast_reroute(self, congested_zone: str, divert_to_zone: str):
        """Updates signage to redirect crowds away from a congested choke point."""
        for screen in self.active_messages:
            if congested_zone in screen["location"]:
                screen["text"] = f"⚠️ HIGH TRAFFIC AHEAD: Please proceed to {divert_to_zone} for faster entry."
                screen["color"] = "ORANGE"

    def get_screens(self) -> List[Dict]:
        return self.active_messages

