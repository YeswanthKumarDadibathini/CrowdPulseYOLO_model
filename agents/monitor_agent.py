"""
CrowdPulse - Monitor Agent (Phase 4)
Project: CrowdPulse - Crowd Density Management System (SIH25165)
Phase: Phase 4 Multi-Agent Coordination

Beginner-Friendly Explanation:
The Monitor Agent acts as the 'Eyes' of the multi-agent system.
Its job is to:
1. Continuously measure crowd headcounts from camera feeds or detectors
2. Track the rate of change (how fast people are entering or leaving per minute)
3. Package raw numbers into a clean 'ZoneTelemetry' report
4. Send the telemetry to the Anomaly Detection Agent for safety checks.
"""

from collections import deque
from datetime import datetime
from typing import Dict, Optional, Tuple
import numpy as np

import config
from agents.schemas import AgentMessage, ZoneTelemetry, utc_now


class MonitorAgent:
    """Agent responsible for observing zone telemetry and tracking velocity/surge rates."""

    def __init__(self, agent_id: str = "MonitorAgent-01"):
        self.agent_id = agent_id
        # Stores recent (timestamp, count) tuples for each zone to compute surge velocity
        self.history: Dict[str, deque] = {}

    def record_reading(
        self,
        zone_id: str,
        current_count: int,
        capacity_limit: Optional[int] = None,
        bottleneck_detected: bool = False,
        hotspot_count: int = 0,
        inference_latency_ms: float = 0.0,
        timestamp: Optional[datetime] = None,
    ) -> Tuple[ZoneTelemetry, AgentMessage]:
        """Processes a new headcount reading and calculates the surge rate per minute."""
        if timestamp is None:
            timestamp = utc_now()

        # Initialize history queue for this zone if new
        if zone_id not in self.history:
            self.history[zone_id] = deque(maxlen=20)

        zone_cfg = config.get_zone_by_id(zone_id)
        zone_name = zone_cfg["name"] if zone_cfg else zone_id
        if capacity_limit is None:
            capacity_limit = zone_cfg["capacity_limit"] if zone_cfg else 100

        # Calculate rate of change (people per minute)
        surge_rate_per_min = 0.0
        if len(self.history[zone_id]) > 0:
            prev_time, prev_count = self.history[zone_id][-1]
            time_diff_sec = max((timestamp - prev_time).total_seconds(), 0.5)
            count_diff = current_count - prev_count
            surge_rate_per_min = round((count_diff / time_diff_sec) * 60.0, 1)

        # Store in rolling history
        self.history[zone_id].append((timestamp, current_count))

        occupancy_pct = round((current_count / max(capacity_limit, 1)) * 100.0, 1)

        # Package telemetry
        telemetry = ZoneTelemetry(
            zone_id=zone_id,
            zone_name=zone_name,
            timestamp=timestamp,
            current_count=current_count,
            capacity_limit=capacity_limit,
            occupancy_pct=occupancy_pct,
            surge_rate_per_min=surge_rate_per_min,
            bottleneck_detected=bottleneck_detected,
            hotspot_count=hotspot_count,
            inference_latency_ms=inference_latency_ms,
        )

        message = AgentMessage(
            sender=self.agent_id,
            recipient="AnomalyAgent",
            message_type="TELEMETRY",
            content=telemetry.to_dict(),
            timestamp=timestamp,
        )

        return telemetry, message

