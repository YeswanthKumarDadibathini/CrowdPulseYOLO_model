"""
CrowdPulse - Vision & Detection Unit Tests (Phase 2)
Project: CrowdPulse - Crowd Density Management System (SIH25165)
Phase: Phase 2 Vision & Detection Pipeline

Run using:
pytest tests/test_detection.py -v
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

import config
from detection.dataset_loader import (
    get_available_datasets,
    get_dataset_metadata,
    get_dataset_summary,
    load_dataset_sample,
)
from detection.detector import CrowdDetector
from detection.video_stream import VideoStreamProcessor, generate_synthetic_demo_video


class TestDatasetLoader:
    """Verifies that the uploaded parquet crowd datasets can be discovered and parsed."""

    def test_get_available_datasets(self):
        datasets = get_available_datasets()
        assert isinstance(datasets, dict)
        assert len(datasets) == 4, f"Expected 4 parquet datasets, found {len(datasets)}"

    def test_get_dataset_summary(self):
        summary = get_dataset_summary()
        assert not summary.empty
        assert "Total Images" in summary.columns
        assert summary["Total Images"].sum() == 1198

    def test_load_sample_image(self):
        datasets = get_available_datasets()
        first_path = list(datasets.values())[0]

        img_rgb, meta = load_dataset_sample(first_path, index=0)
        assert isinstance(img_rgb, np.ndarray)
        assert len(img_rgb.shape) == 3
        assert img_rgb.shape[2] == 3  # RGB
        assert "ground_truth_count" in meta
        assert meta["ground_truth_count"] >= 0
        assert "file_name" in meta


class TestCrowdDetector:
    """Verifies YOLO model loading, person detection, and density classification."""

    @pytest.fixture
    def detector(self):
        return CrowdDetector()

    def test_detector_initialization(self, detector):
        assert detector.model is not None
        assert detector.device in ["cuda", "cpu"]

    def test_classify_density(self, detector):
        # With capacity 100:
        # < 60 is NORMAL
        level, badge, color = detector.classify_density(count=30, capacity_limit=100)
        assert level == "NORMAL"
        assert "Safe" in badge

        # 60 - 80 is MODERATE
        level, badge, color = detector.classify_density(count=70, capacity_limit=100)
        assert level == "MODERATE"
        assert "Caution" in badge

        # > 80 is CRITICAL
        level, badge, color = detector.classify_density(count=95, capacity_limit=100)
        assert level == "CRITICAL"
        assert "Overcrowded" in badge

    def test_detect_people_output_schema(self, detector):
        # Create a synthetic 300x400 image
        dummy_img = np.zeros((300, 400, 3), dtype=np.uint8)
        result = detector.detect_people(
            image_rgb=dummy_img,
            confidence_threshold=0.25,
            capacity_limit=50,
            draw_annotations=True
        )

        assert isinstance(result, dict)
        assert "count" in result
        assert "boxes" in result
        assert "density_level" in result
        assert "risk_badge" in result
        assert "occupancy_pct" in result
        assert "inference_time_ms" in result
        assert "annotated_image" in result
        assert result["annotated_image"].shape == dummy_img.shape
        assert result["count"] >= 0

    def test_high_density_sliced_mode(self, detector):
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_people(
            image_rgb=dummy_img,
            confidence_threshold=0.15,
            mode="high_density_sliced",
            draw_annotations=True
        )
        assert result["mode_used"] == "high_density_sliced"
        assert result["count"] >= 0

    def test_calibrated_density_mode(self, detector):
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_people(
            image_rgb=dummy_img,
            confidence_threshold=0.15,
            mode="calibrated",
            draw_annotations=True
        )
        assert result["mode_used"] == "calibrated"
        assert isinstance(result["count"], int)


class TestVideoStream:
    """Verifies video creation and streaming frame processing."""

    def test_generate_synthetic_video(self, tmp_path):
        video_out = tmp_path / "test_sim.mp4"
        generated = generate_synthetic_demo_video(output_path=video_out, num_frames=10, fps=10)
        assert generated.exists()
        assert generated.stat().st_size > 0

    def test_process_video_frames(self, tmp_path):
        video_out = tmp_path / "test_sim.mp4"
        generate_synthetic_demo_video(output_path=video_out, num_frames=10, fps=10)

        processor = VideoStreamProcessor()
        frames_yielded = 0
        for frame, stats in processor.process_video_file(video_out, max_frames=3):
            frames_yielded += 1
            assert isinstance(frame, np.ndarray)
            assert "current_count" in stats
            assert "current_fps" in stats

        assert frames_yielded == 3
