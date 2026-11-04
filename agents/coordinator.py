"""
CrowdPulse - Multi-Agent Coordinator (Phase 4)
Project: CrowdPulse - Crowd Density Management System (SIH25165)
Phase: Phase 4 Multi-Agent Coordination

Beginner-Friendly Explanation:
The MultiAgentCoordinator is the 'Conductor' of the multi-agent system.
It ties together:
1. MonitorAgent (Measures numbers and velocity)
2. AnomalyDetectionAgent (Identifies danger/stampede patterns)
3. DispatchAgent (Formulates on-ground countermeasures)

It maintains an event bus log showing how agents pass messages to each other
and can simulate dynamic crowd events (Surges, Bottlenecks, Evacuations).
"""

from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import config
from agents.anomaly_agent import AnomalyDetectionAgent
from agents.dispatch_agent import DispatchAgent
from agents.monitor_agent import MonitorAgent
from agents.schemas import AgentMessage, AnomalyReport, DispatchAction, ZoneTelemetry, utc_now
from database.db_service import log_density_reading


class MultiAgentCoordinator:
    """Orchestrates communication and lifecycle between all CrowdPulse intelligent agents."""

    def __init__(self):
        self.monitor_agent = MonitorAgent(agent_id="MonitorAgent")
        self.anomaly_agent = AnomalyDetectionAgent(agent_id="AnomalyAgent")
        self.dispatch_agent = DispatchAgent(agent_id="DispatchAgent")

        # Central Message Bus Log (FIFO deque of recent communications)
        self.message_bus: deque = deque(maxlen=100)

        # Log system boot message
        boot_msg = AgentMessage(
            sender="Coordinator",
            recipient="BROADCAST",
            message_type="STATUS_UPDATE",
            content={"status": "SYSTEM_ONLINE", "message": "Multi-agent coordination bus initialized."},
            timestamp=utc_now(),
        )
        self.message_bus.append(boot_msg)

    def process_zone_update(
        self,
        zone_id: str,
        current_count: int,
        capacity_limit: Optional[int] = None,
        bottleneck_detected: bool = False,
        hotspot_count: int = 0,
        inference_latency_ms: float = 0.0,
        timestamp: Optional[datetime] = None,
    ) -> Dict:
        """Executes one multi-agent cycle for a zone:

        Monitor -> Anomaly Detector -> Dispatcher -> Database Logging.
        """
        if timestamp is None:
            timestamp = utc_now()

        # Step 1: Monitor Agent calculates telemetry & velocity
        telemetry, msg_telemetry = self.monitor_agent.record_reading(
            zone_id=zone_id,
            current_count=current_count,
            capacity_limit=capacity_limit,
            bottleneck_detected=bottleneck_detected,
            hotspot_count=hotspot_count,
            inference_latency_ms=inference_latency_ms,
            timestamp=timestamp,
        )
        self.message_bus.append(msg_telemetry)

        # Step 2: Anomaly Agent analyzes telemetry
        anomaly_report, msg_anomaly = self.anomaly_agent.evaluate_telemetry(telemetry)
        if msg_anomaly:
            self.message_bus.append(msg_anomaly)

        # Step 3: Dispatch Agent formulates tactical response if anomaly detected
        actions = []
        if anomaly_report is not None:
            actions, action_messages = self.dispatch_agent.formulate_response(anomaly_report)
            for m in action_messages:
                self.message_bus.append(m)

        # Step 4: Persist reading and alerts in SQLite database
        db_record = log_density_reading(
            zone_id=zone_id,
            person_count=current_count,
            capacity_limit=telemetry.capacity_limit,
            bottleneck_detected=bottleneck_detected,
            inference_time_ms=inference_latency_ms,
            timestamp=timestamp,
        )

        return {
            "telemetry": telemetry.to_dict(),
            "anomaly": anomaly_report.to_dict() if anomaly_report else None,
            "actions": [a.to_dict() for a in actions],
            "database_logged": db_record,
        }

    def simulate_crowd_scenario(self, scenario: str = "SURGE", zone_id: str = "zone_1") -> Dict:
        """Simulates synthetic real-world crowd events to demonstrate multi-agent responses."""
        zone_cfg = config.get_zone_by_id(zone_id) or config.DEFAULT_ZONES[0]
        capacity = zone_cfg["capacity_limit"]

        if scenario == "SURGE":
            # Rapid surge entering in 30 seconds: count jumps significantly
            # First reading at base
            self.process_zone_update(zone_id=zone_id, current_count=int(capacity * 0.4))
            # Second reading surging rapidly
            surge_count = int(capacity * 0.95)
            result = self.process_zone_update(
                zone_id=zone_id,
                current_count=surge_count,
                capacity_limit=capacity,
                bottleneck_detected=False,
            )
            scenario_desc = f"Simulated Sudden Crowd Surge at {zone_cfg['name']}: {surge_count} people (+{surge_count - int(capacity * 0.4)} influx)."

        elif scenario == "BOTTLENECK":
            # Spatial bottleneck cluster at turnstiles
            count = int(capacity * 0.75)
            result = self.process_zone_update(
                zone_id=zone_id,
                current_count=count,
                capacity_limit=capacity,
                bottleneck_detected=True,
                hotspot_count=2,
            )
            scenario_desc = f"Simulated Physical Bottleneck at {zone_cfg['name']}: Dense choke point clustering in 2 grid cells."

        elif scenario == "CRITICAL_OVERCROWDING":
            # Over 100% capacity
            over_count = int(capacity * 1.15)
            result = self.process_zone_update(
                zone_id=zone_id,
                current_count=over_count,
                capacity_limit=capacity,
                bottleneck_detected=True,
                hotspot_count=3,
            )
            scenario_desc = f"Simulated Dangerous Overcrowding at {zone_cfg['name']}: {over_count}/{capacity} people (115% capacity)."

        else:
            # Normal safe condition
            normal_count = int(capacity * 0.35)
            result = self.process_zone_update(
                zone_id=zone_id,
                current_count=normal_count,
                capacity_limit=capacity,
                bottleneck_detected=False,
            )
            scenario_desc = f"Simulated Normal Operation at {zone_cfg['name']}: {normal_count} people (Safe, 35% capacity)."

        result["scenario_description"] = scenario_desc
        return result

    def get_recent_messages(self, limit: int = 25) -> List[Dict]:
        """Returns the most recent messages from the inter-agent bus."""
        msgs = list(self.message_bus)[-limit:]
        return [m.to_dict() for m in reversed(msgs)]

