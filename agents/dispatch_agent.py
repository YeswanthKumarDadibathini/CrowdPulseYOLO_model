"""
CrowdPulse - Dispatch & Response Agent (Phase 4)
Project: CrowdPulse - Crowd Density Management System (SIH25165)
Phase: Phase 4 Multi-Agent Coordination

Beginner-Friendly Explanation:
The Dispatch Agent acts as the 'Incident Commander'.
When it receives an AnomalyReport, it formulates an action plan:
1. Deploys On-Ground Security Teams / Crowd Marshals
2. Updates Digital Signage displays & PA audio systems to divert walking traffic
3. Modifies Electronic Turnstile gates (throttles inflow or opens emergency bypasses)

It keeps track of all issued action items and their status (Pending -> Executed -> Resolved).
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
import uuid

from agents.schemas import AgentMessage, AnomalyReport, DispatchAction, utc_now


class DispatchAgent:
    """Agent responsible for formulating tactical responses and dispatching emergency countermeasures."""

    def __init__(self, agent_id: str = "DispatchAgent-01"):
        self.agent_id = agent_id
        # In-memory history of issued dispatch actions
        self.action_history: List[DispatchAction] = []

    def formulate_response(
        self,
        anomaly: AnomalyReport
    ) -> Tuple[List[DispatchAction], List[AgentMessage]]:
        """Analyzes an anomaly report and generates a list of coordinated dispatch actions."""
        actions: List[DispatchAction] = []
        messages: List[AgentMessage] = []

        # 1. Responses for Critical Overcrowding (Capacity Breached)
        if anomaly.anomaly_type == "CRITICAL_OVERCROWDING":
            act_id1 = f"ACT-{uuid.uuid4().hex[:6].upper()}"
            action_security = DispatchAction(
                action_id=act_id1,
                anomaly_id=anomaly.anomaly_id,
                zone_id=anomaly.zone_id,
                timestamp=anomaly.timestamp,
                action_type="DISPATCH_SECURITY",
                target_team="Rapid Response Marshal Squad Alpha",
                priority="URGENT",
                instructions=(
                    f"Immediate physical crowd control required at {anomaly.zone_name}. "
                    f"Zone at {anomaly.metrics.get('occupancy_pct', 100)}% capacity. "
                    f"Establish physical cordons and assist evacuation."
                ),
                status="EXECUTED",
            )
            actions.append(action_security)

            act_id2 = f"ACT-{uuid.uuid4().hex[:6].upper()}"
            action_gate = DispatchAction(
                action_id=act_id2,
                anomaly_id=anomaly.anomaly_id,
                zone_id=anomaly.zone_id,
                timestamp=anomaly.timestamp,
                action_type="RESTRICT_TURNSTILES",
                target_team="Automated Gate Control System",
                priority="URGENT",
                instructions=f"Engage temporary inflow lock on entry turnstiles at {anomaly.zone_name}. Halt new entries.",
                status="EXECUTED",
            )
            actions.append(action_gate)

            act_id3 = f"ACT-{uuid.uuid4().hex[:6].upper()}"
            action_pa = DispatchAction(
                action_id=act_id3,
                anomaly_id=anomaly.anomaly_id,
                zone_id=anomaly.zone_id,
                timestamp=anomaly.timestamp,
                action_type="BROADCAST_AUDIO",
                target_team="Public Address & Digital Signage",
                priority="HIGH",
                instructions=f"Broadcast Evacuation / Divert Announcement: 'Zone {anomaly.zone_name} congested. Please use alternate corridors.'",
                status="EXECUTED",
            )
            actions.append(action_pa)

        # 2. Responses for Sudden Crowd Surge
        elif anomaly.anomaly_type == "SUDDEN_SURGE":
            act_id = f"ACT-{uuid.uuid4().hex[:6].upper()}"
            action_surge = DispatchAction(
                action_id=act_id,
                anomaly_id=anomaly.anomaly_id,
                zone_id=anomaly.zone_id,
                timestamp=anomaly.timestamp,
                action_type="METER_TURNSTILES",
                target_team="Automated Turnstile Manager",
                priority="HIGH",
                instructions=(
                    f"Throttle entry speed at {anomaly.zone_name} to 10 persons/min. "
                    f"Inflow surge of +{anomaly.metrics.get('surge_rate_per_min', 20)}/min detected."
                ),
                status="EXECUTED",
            )
            actions.append(action_surge)

        # 3. Responses for Static Bottleneck
        elif anomaly.anomaly_type == "STATIC_BOTTLENECK":
            act_id = f"ACT-{uuid.uuid4().hex[:6].upper()}"
            action_bottle = DispatchAction(
                action_id=act_id,
                anomaly_id=anomaly.anomaly_id,
                zone_id=anomaly.zone_id,
                timestamp=anomaly.timestamp,
                action_type="DISPATCH_SECURITY",
                target_team="Crowd Flow Ushers Team Bravo",
                priority="HIGH",
                instructions=(
                    f"Deploy 4 ushers with megaphones to {anomaly.zone_name} to clear localized "
                    f"choke point clustering and restore directional crowd momentum."
                ),
                status="EXECUTED",
            )
            actions.append(action_bottle)

        # 4. Responses for Elevated Density Warning
        else:
            act_id = f"ACT-{uuid.uuid4().hex[:6].upper()}"
            action_sign = DispatchAction(
                action_id=act_id,
                anomaly_id=anomaly.anomaly_id,
                zone_id=anomaly.zone_id,
                timestamp=anomaly.timestamp,
                action_type="UPDATE_SIGNAGE",
                target_team="Wayfinding Digital Screens",
                priority="NORMAL",
                instructions=f"Update overhead LED guidance displays to suggest adjacent less crowded zones.",
                status="EXECUTED",
            )
            actions.append(action_sign)

        # Store in agent history
        self.action_history.extend(actions)

        # Build messages for each action
        for act in actions:
            messages.append(
                AgentMessage(
                    sender=self.agent_id,
                    recipient=act.target_team,
                    message_type="ACTION_ORDER",
                    content=act.to_dict(),
                    timestamp=anomaly.timestamp,
                )
            )

        return actions, messages
