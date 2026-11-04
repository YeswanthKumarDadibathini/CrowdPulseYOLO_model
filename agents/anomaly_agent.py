"""
CrowdPulse - Anomaly Detection Agent (Phase 4)
Project: CrowdPulse - Crowd Density Management System (SIH25165)
Phase: Phase 4 Multi-Agent Coordination

Beginner-Friendly Explanation:
The Anomaly Agent acts as the 'Security Analyst'.
It inspects telemetry incoming from the Monitor Agent and applies safety logic
to detect dangerous patterns before an incident or stampede occurs:
1. Critical Overcrowding (Capacity > 100%)
2. Sudden Crowd Surge (Rapid inflow rate of people)
3. Stationary Bottlenecks (Clusters choking an exit or gate)

If an anomaly is detected, it generates an AnomalyReport and forwards it
to the Dispatch Agent for action.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
import uuid

import config
from agents.schemas import AgentMessage, AnomalyReport, ZoneTelemetry, utc_now


class AnomalyDetectionAgent:
    """Agent responsible for inspecting zone telemetry and classifying crowd risks."""

    def __init__(self, agent_id: str = "AnomalyAgent-01"):
        self.agent_id = agent_id
        # Threshold for rapid influx of people per minute
        self.surge_threshold_per_min = 20.0

    def evaluate_telemetry(
        self,
        telemetry: ZoneTelemetry
    ) -> Tuple[Optional[AnomalyReport], Optional[AgentMessage]]:
        """Evaluates telemetry against risk rules and returns an AnomalyReport if triggered."""
        anomaly = None

        # Rule 1: Critical Overcrowding (Over safe capacity)
        if telemetry.occupancy_pct >= 100.0:
            anomaly = AnomalyReport(
                anomaly_id=f"ANOM-{uuid.uuid4().hex[:6].upper()}",
                zone_id=telemetry.zone_id,
                zone_name=telemetry.zone_name,
                timestamp=telemetry.timestamp,
                anomaly_type="CRITICAL_OVERCROWDING",
                severity="CRITICAL",
                confidence=0.98,
                description=(
                    f"Zone capacity breached! Current occupancy is {telemetry.occupancy_pct}% "
                    f"({telemetry.current_count}/{telemetry.capacity_limit} persons). High stampede risk."
                ),
                metrics={
                    "current_count": telemetry.current_count,
                    "capacity_limit": telemetry.capacity_limit,
                    "occupancy_pct": telemetry.occupancy_pct,
                },
            )

        # Rule 2: Sudden Crowd Surge (Rapid rate of change)
        elif telemetry.surge_rate_per_min >= self.surge_threshold_per_min:
            confidence = min(0.95, 0.70 + (telemetry.surge_rate_per_min / 100.0))
            anomaly = AnomalyReport(
                anomaly_id=f"ANOM-{uuid.uuid4().hex[:6].upper()}",
                zone_id=telemetry.zone_id,
                zone_name=telemetry.zone_name,
                timestamp=telemetry.timestamp,
                anomaly_type="SUDDEN_SURGE",
                severity="HIGH",
                confidence=round(confidence, 2),
                description=(
                    f"Rapid crowd influx detected! Surge rate is +{telemetry.surge_rate_per_min} people/min. "
                    f"Occupancy rising rapidly at {telemetry.occupancy_pct}%."
                ),
                metrics={
                    "surge_rate_per_min": telemetry.surge_rate_per_min,
                    "current_count": telemetry.current_count,
                },
            )

        # Rule 3: Stationary Bottleneck Cluster
        elif telemetry.bottleneck_detected:
            severity = "HIGH" if telemetry.occupancy_pct >= 75.0 else "MEDIUM"
            anomaly = AnomalyReport(
                anomaly_id=f"ANOM-{uuid.uuid4().hex[:6].upper()}",
                zone_id=telemetry.zone_id,
                zone_name=telemetry.zone_name,
                timestamp=telemetry.timestamp,
                anomaly_type="STATIC_BOTTLENECK",
                severity=severity,
                confidence=0.88,
                description=(
                    f"Spatial bottleneck detected! Dense clustering observed in {telemetry.hotspot_count} "
                    f"localized grid cell(s). Crowd flow is physically obstructed."
                ),
                metrics={
                    "hotspots": telemetry.hotspot_count,
                    "occupancy_pct": telemetry.occupancy_pct,
                },
            )

        # Rule 4: Moderate Elevated Density Warning
        elif telemetry.occupancy_pct >= config.DENSITY_THRESHOLDS["MODERATE"]:
            anomaly = AnomalyReport(
                anomaly_id=f"ANOM-{uuid.uuid4().hex[:6].upper()}",
                zone_id=telemetry.zone_id,
                zone_name=telemetry.zone_name,
                timestamp=telemetry.timestamp,
                anomaly_type="ELEVATED_DENSITY",
                severity="MEDIUM",
                confidence=0.80,
                description=(
                    f"Caution: Zone is approaching high density ({telemetry.occupancy_pct}% of capacity). "
                    f"Close monitoring advised."
                ),
                metrics={"occupancy_pct": telemetry.occupancy_pct},
            )

        # If an anomaly was detected, package an inter-agent message
        message = None
        if anomaly is not None:
            message = AgentMessage(
                sender=self.agent_id,
                recipient="DispatchAgent",
                message_type="ANOMALY_ALERT",
                content=anomaly.to_dict(),
                timestamp=telemetry.timestamp,
            )

        return anomaly, message

