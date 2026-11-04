"""
CrowdPulse - Multi-Agent System Unit Tests (Phase 4)
Project: CrowdPulse - Crowd Density Management System (SIH25165)
Phase: Phase 4 Multi-Agent Coordination

Run using:
pytest tests/test_agents.py -v
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from agents.anomaly_agent import AnomalyDetectionAgent
from agents.coordinator import MultiAgentCoordinator
from agents.dispatch_agent import DispatchAgent
from agents.monitor_agent import MonitorAgent
from agents.schemas import AnomalyReport, ZoneTelemetry, utc_now
from services.dispatch_service import (
    DigitalSignageService,
    MarshalSquadTracker,
    TurnstileGateController,
)


class TestMonitorAgent:
    """Verifies that the Monitor Agent calculates occupancy and rate-of-change accurately."""

    def test_record_initial_reading(self):
        agent = MonitorAgent()
        telemetry, msg = agent.record_reading(
            zone_id="zone_1",
            current_count=50,
            capacity_limit=100,
        )
        assert telemetry.current_count == 50
        assert telemetry.occupancy_pct == 50.0
        assert telemetry.surge_rate_per_min == 0.0
        assert msg.message_type == "TELEMETRY"

    def test_surge_rate_calculation(self):
        agent = MonitorAgent()
        t0 = utc_now()
        # Initial reading: 20 people at t0
        agent.record_reading(zone_id="zone_1", current_count=20, capacity_limit=100, timestamp=t0)

        # Second reading: 50 people after 30 seconds (+30 people in 30 sec = +60 people/min)
        t1 = t0 + timedelta(seconds=30)
        telemetry, _ = agent.record_reading(zone_id="zone_1", current_count=50, capacity_limit=100, timestamp=t1)

        assert telemetry.surge_rate_per_min == 60.0


class TestAnomalyAgent:
    """Verifies that safety anomaly rules trigger appropriately based on crowd patterns."""

    @pytest.fixture
    def anomaly_agent(self):
        return AnomalyDetectionAgent()

    def test_critical_overcrowding_anomaly(self, anomaly_agent):
        telemetry = ZoneTelemetry(
            zone_id="zone_1",
            zone_name="North Gate",
            timestamp=utc_now(),
            current_count=165,
            capacity_limit=150,
            occupancy_pct=110.0,
        )
        report, msg = anomaly_agent.evaluate_telemetry(telemetry)
        assert report is not None
        assert report.anomaly_type == "CRITICAL_OVERCROWDING"
        assert report.severity == "CRITICAL"
        assert msg.recipient == "DispatchAgent"

    def test_sudden_surge_anomaly(self, anomaly_agent):
        telemetry = ZoneTelemetry(
            zone_id="zone_2",
            zone_name="Central Plaza",
            timestamp=utc_now(),
            current_count=300,
            capacity_limit=500,
            occupancy_pct=60.0,
            surge_rate_per_min=35.0,  # Rapid influx
        )
        report, msg = anomaly_agent.evaluate_telemetry(telemetry)
        assert report is not None
        assert report.anomaly_type == "SUDDEN_SURGE"
        assert report.severity == "HIGH"

    def test_static_bottleneck_anomaly(self, anomaly_agent):
        telemetry = ZoneTelemetry(
            zone_id="zone_3",
            zone_name="Food Concourse",
            timestamp=utc_now(),
            current_count=180,
            capacity_limit=250,
            occupancy_pct=72.0,
            bottleneck_detected=True,
            hotspot_count=2,
        )
        report, msg = anomaly_agent.evaluate_telemetry(telemetry)
        assert report is not None
        assert report.anomaly_type == "STATIC_BOTTLENECK"
        assert report.severity == "MEDIUM"

    def test_normal_telemetry_no_anomaly(self, anomaly_agent):
        telemetry = ZoneTelemetry(
            zone_id="zone_4",
            zone_name="South Exit",
            timestamp=utc_now(),
            current_count=40,
            capacity_limit=200,
            occupancy_pct=20.0,
            surge_rate_per_min=2.0,
            bottleneck_detected=False,
        )
        report, msg = anomaly_agent.evaluate_telemetry(telemetry)
        assert report is None
        assert msg is None


class TestDispatchAgent:
    """Verifies that the Dispatch Agent generates tactical countermeasures for anomalies."""

    @pytest.fixture
    def dispatch_agent(self):
        return DispatchAgent()

    def test_critical_overcrowding_countermeasures(self, dispatch_agent):
        anomaly = AnomalyReport(
            anomaly_id="ANOM-TEST1",
            zone_id="zone_1",
            zone_name="North Gate",
            timestamp=utc_now(),
            anomaly_type="CRITICAL_OVERCROWDING",
            severity="CRITICAL",
            confidence=0.98,
            description="Zone at 110% capacity",
            metrics={"occupancy_pct": 110.0},
        )
        actions, messages = dispatch_agent.formulate_response(anomaly)
        assert len(actions) == 3
        action_types = [a.action_type for a in actions]
        assert "DISPATCH_SECURITY" in action_types
        assert "RESTRICT_TURNSTILES" in action_types
        assert "BROADCAST_AUDIO" in action_types

    def test_sudden_surge_countermeasures(self, dispatch_agent):
        anomaly = AnomalyReport(
            anomaly_id="ANOM-TEST2",
            zone_id="zone_2",
            zone_name="Central Plaza",
            timestamp=utc_now(),
            anomaly_type="SUDDEN_SURGE",
            severity="HIGH",
            confidence=0.85,
            description="Rapid crowd influx",
            metrics={"surge_rate_per_min": 40.0},
        )
        actions, messages = dispatch_agent.formulate_response(anomaly)
        assert len(actions) == 1
        assert actions[0].action_type == "METER_TURNSTILES"
        assert actions[0].priority == "HIGH"


class TestMultiAgentCoordinator:
    """Verifies end-to-end multi-agent orchestration and scenario simulations."""

    def test_coordinator_initialization(self):
        coord = MultiAgentCoordinator()
        msgs = coord.get_recent_messages()
        assert len(msgs) >= 1
        assert msgs[0]["content"]["status"] == "SYSTEM_ONLINE"

    def test_coordinator_full_cycle(self):
        coord = MultiAgentCoordinator()
        result = coord.process_zone_update(
            zone_id="zone_1",
            current_count=180,  # Capacity is 150 -> Overcrowding
            bottleneck_detected=True,
        )
        assert "telemetry" in result
        assert "anomaly" in result
        assert result["anomaly"] is not None
        assert "actions" in result
        assert len(result["actions"]) > 0
        assert "database_logged" in result

    def test_simulate_crowd_scenario(self):
        coord = MultiAgentCoordinator()
        surge_res = coord.simulate_crowd_scenario(scenario="SURGE", zone_id="zone_1")
        assert "scenario_description" in surge_res
        assert "telemetry" in surge_res


class TestDispatchServices:
    """Verifies that physical field equipment trackers operate properly."""

    def test_marshal_squad_tracker(self):
        tracker = MarshalSquadTracker()
        squads = tracker.get_all_squads()
        assert len(squads) >= 3

        dispatched = tracker.dispatch_squad("SQUAD-ALPHA", "North Gate", "Stampede Prevention")
        assert dispatched["status"] == "DISPATCHED"
        assert dispatched["current_zone"] == "North Gate"

    def test_turnstile_gate_controller(self):
        controller = TurnstileGateController()
        gates = controller.get_all_gates()
        assert len(gates) == 4

        updated = controller.set_gate_mode("zone_1", "LOCKED_INFLOW", rate_limit_ppm=0)
        assert updated["gate_status"] == "LOCKED_INFLOW"
        assert updated["flow_rate_limit_ppm"] == 0

    def test_digital_signage_service(self):
        signage = DigitalSignageService()
        screens = signage.get_screens()
        assert len(screens) >= 3

        signage.broadcast_reroute("North Entrance", "South Entrance")
        updated_screens = signage.get_screens()
        north_screen = next(s for s in updated_screens if s["location"] == "North Entrance")
        assert "HIGH TRAFFIC AHEAD" in north_screen["text"]

