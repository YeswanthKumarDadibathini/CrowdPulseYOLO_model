"""
CrowdPulse - Spatial Analytics & Database Unit Tests (Phase 3)
Project: CrowdPulse - Crowd Density Management System (SIH25165)
Phase: Phase 3 Spatial Analytics & Incident Database

Run using:
pytest tests/test_phase3.py -v
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from database.db_service import (
    get_active_alerts,
    get_db_session,
    get_recent_logs_dataframe,
    init_db,
    log_density_reading,
    seed_mock_history_if_empty,
    update_alert_status,
)
from database.models import DensityLog, IncidentAlert, Zone
from detection.heatmap import SpatialHeatmapGenerator


class TestDatabase:
    """Verifies SQLite ORM database initialization, logging, alerts, and querying."""

    @pytest.fixture(autouse=True)
    def setup_database(self):
        init_db()

    def test_database_tables_and_zones_seeded(self):
        with get_db_session() as session:
            zones = session.query(Zone).all()
            assert len(zones) >= 4
            zone_ids = [z.id for z in zones]
            assert "zone_1" in zone_ids
            assert "zone_2" in zone_ids

    def test_log_density_reading_normal(self):
        res = log_density_reading(
            zone_id="zone_1",
            person_count=30,  # Capacity is 150 -> 20% occupancy
            bottleneck_detected=False,
            inference_time_ms=32.5,
        )
        assert res["risk_level"] == "NORMAL"
        assert res["occupancy_pct"] == 20.0
        assert res["alert_created"] is False

    def test_log_density_reading_critical_creates_alert(self):
        res = log_density_reading(
            zone_id="zone_1",
            person_count=165,  # Capacity 150 -> 110% occupancy
            bottleneck_detected=False,
            inference_time_ms=35.0,
        )
        assert res["risk_level"] == "CRITICAL"
        assert res["occupancy_pct"] == 110.0
        assert res["alert_created"] is True

        # Verify active alerts list contains the alert
        active_alerts = get_active_alerts()
        assert len(active_alerts) > 0
        latest_alert = active_alerts[0]
        assert latest_alert["severity"] in ["HIGH", "CRITICAL"]

    def test_update_alert_status(self):
        # Create an alert
        log_density_reading(zone_id="zone_2", person_count=600)  # Over capacity 500
        alerts = get_active_alerts()
        assert len(alerts) > 0

        target_id = alerts[0]["id"]
        update_alert_status(target_id, "RESOLVED")

        # Verify it was resolved
        with get_db_session() as session:
            resolved = session.query(IncidentAlert).filter(IncidentAlert.id == target_id).first()
            assert resolved.status == "RESOLVED"
            assert resolved.resolved_at is not None

    def test_get_recent_logs_dataframe(self):
        df = get_recent_logs_dataframe(limit=10)
        assert not df.empty
        assert "Zone Name" in df.columns
        assert "Count" in df.columns
        assert "Occupancy (%)" in df.columns

    def test_seed_mock_history_if_empty(self):
        seed_mock_history_if_empty(num_hours=2)
        with get_db_session() as session:
            assert session.query(DensityLog).count() > 0


class TestSpatialHeatmap:
    """Verifies Gaussian density heatmap synthesis and spatial grid bottleneck clustering."""

    @pytest.fixture
    def heatmap_gen(self):
        return SpatialHeatmapGenerator(grid_rows=4, grid_cols=4)

    def test_generate_heatmap_with_detections(self, heatmap_gen):
        # Create dummy image: 400 height x 600 width x 3
        img = np.full((400, 600, 3), 120, dtype=np.uint8)
        boxes = [
            {"bbox": [100, 100, 140, 200], "confidence": 0.85},
            {"bbox": [105, 110, 145, 210], "confidence": 0.88},
            {"bbox": [110, 115, 150, 215], "confidence": 0.90},
            {"bbox": [400, 300, 440, 380], "confidence": 0.82},
        ]

        blended, raw_heat = heatmap_gen.generate_heatmap(img, boxes)
        assert blended.shape == img.shape
        assert blended.dtype == np.uint8
        assert raw_heat.shape == img.shape

    def test_generate_heatmap_empty_boxes(self, heatmap_gen):
        img = np.full((300, 300, 3), 100, dtype=np.uint8)
        blended, raw_heat = heatmap_gen.generate_heatmap(img, [])
        assert blended.shape == img.shape
        # Should not crash and return valid image

    def test_analyze_spatial_grid_and_bottlenecks(self, heatmap_gen):
        img = np.full((400, 400, 3), 50, dtype=np.uint8)
        # Cluster 6 people into cell [0, 0] (x < 100, y < 100)
        boxes = [
            {"bbox": [20, 20, 40, 80], "confidence": 0.9},
            {"bbox": [30, 25, 50, 85], "confidence": 0.9},
            {"bbox": [40, 30, 60, 90], "confidence": 0.9},
            {"bbox": [25, 40, 45, 95], "confidence": 0.9},
            {"bbox": [35, 45, 55, 95], "confidence": 0.9},
            {"bbox": [45, 50, 65, 98], "confidence": 0.9},
        ]

        analysis = heatmap_gen.analyze_spatial_grid(img, boxes, bottleneck_threshold_pct=30.0)
        assert "grid_counts" in analysis
        assert "hotspots" in analysis
        assert analysis["bottleneck_detected"] is True
        assert len(analysis["hotspots"]) >= 1
        assert analysis["hotspots"][0]["row"] == 0
        assert analysis["hotspots"][0]["col"] == 0
        assert analysis["grid_annotated_image"].shape == img.shape

    def test_proximity_crush_graph(self, heatmap_gen):
        img = np.full((400, 400, 3), 80, dtype=np.uint8)
        # People standing very close together (< 65 pixels)
        boxes = [
            {"bbox": [100, 100, 120, 160], "confidence": 0.9},
            {"bbox": [115, 105, 135, 165], "confidence": 0.9},
            {"bbox": [110, 120, 130, 180], "confidence": 0.9},
            {"bbox": [125, 115, 145, 175], "confidence": 0.9},
        ]
        annotated, stats = heatmap_gen.generate_proximity_crush_graph(img, boxes, distance_threshold_px=65)
        assert annotated.shape == img.shape
        assert stats["total_connections"] > 0
        assert stats["crush_index"] > 0
        assert "risk_status" in stats

