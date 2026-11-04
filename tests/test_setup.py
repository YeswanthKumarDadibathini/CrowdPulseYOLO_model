"""
CrowdPulse - Setup & Environment Unit Tests (Phase 1)
Project: CrowdPulse - Crowd Density Management System (SIH25165)
Phase: Phase 1 Foundation

Run using:
pytest tests/test_setup.py -v
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import config
from utils.environment_check import get_environment_report, check_nvidia_smi


class TestConfiguration:
    """Verifies that project configuration parameters, paths, and default zones

    are correctly structured.
    """

    def test_project_metadata(self):
        """Test essential project information fields."""
        assert config.PROJECT_NAME == "CrowdPulse"
        assert config.SIH_PROJECT_CODE == "SIH25165"
        assert isinstance(config.APP_VERSION, str)
        assert len(config.APP_VERSION) > 0

    def test_paths_defined_and_exist(self):
        """Test directory paths exist on the filesystem."""
        assert config.BASE_DIR.exists()
        assert config.DATA_DIR.exists()
        assert config.VIDEOS_DIR.exists()
        assert config.DATASETS_DIR.exists()
        assert config.MODELS_DIR.exists()
        assert config.DATABASE_DIR.exists()

    def test_density_thresholds(self):
        """Test density threshold percentages."""
        thresholds = config.DENSITY_THRESHOLDS
        assert "NORMAL" in thresholds
        assert "MODERATE" in thresholds
        assert "CRITICAL" in thresholds
        assert thresholds["NORMAL"] < thresholds["MODERATE"] <= thresholds["CRITICAL"]

    def test_default_zones_structure(self):
        """Test that default zones contain 4 configured demo zones with valid fields."""
        zones = config.DEFAULT_ZONES
        assert isinstance(zones, list)
        assert len(zones) == 4

        required_keys = {"id", "name", "camera_source", "capacity_limit", "current_count", "risk_level"}
        for zone in zones:
            assert required_keys.issubset(zone.keys())
            assert zone["capacity_limit"] > 0
            assert zone["current_count"] >= 0

    def test_get_zone_by_id(self):
        """Test retrieving a zone by ID helper function."""
        zone1 = config.get_zone_by_id("zone_1")
        assert zone1 is not None
        assert zone1["id"] == "zone_1"
        assert "North Gate" in zone1["name"]

        # Non-existent zone
        assert config.get_zone_by_id("non_existent_zone") is None


class TestProjectStructure:
    """Verifies directory layout required by the SIH25165 Phase 1 specification."""

    def test_package_directories_exist(self):
        base = config.BASE_DIR
        assert (base / "agents" / "__init__.py").exists()
        assert (base / "detection" / "__init__.py").exists()
        assert (base / "services" / "__init__.py").exists()
        assert (base / "database" / "__init__.py").exists()
        assert (base / "dashboard" / "__init__.py").exists()
        assert (base / "utils" / "__init__.py").exists()

    def test_gitkeep_files_exist(self):
        base = config.BASE_DIR
        assert (base / "data" / "videos" / ".gitkeep").exists()
        assert (base / "data" / "datasets" / ".gitkeep").exists()
        assert (base / "models" / ".gitkeep").exists()

    def test_gitignore_exists(self):
        base = config.BASE_DIR
        assert (base / ".gitignore").exists()


class TestEnvironmentDiagnostics:
    """Verifies that environment check module runs without crashing

    and yields valid data structures.
    """

    def test_environment_report_schema(self):
        report = get_environment_report()
        assert isinstance(report, dict)
        assert "system" in report
        assert "packages" in report
        assert "pytorch" in report
        assert "nvidia_smi" in report

    def test_system_info(self):
        report = get_environment_report()
        sys_info = report["system"]
        assert "python_version" in sys_info
        assert len(sys_info["python_version"]) > 0
        assert "platform" in sys_info

    def test_packages_reported(self):
        report = get_environment_report()
        pkgs = report["packages"]
        # Required core packages must be tracked
        for pkg in ["streamlit", "numpy", "pandas", "pytest"]:
            assert pkg in pkgs
            assert "installed" in pkgs[pkg]

    def test_pytorch_structure(self):
        report = get_environment_report()
        pytorch = report["pytorch"]
        assert "installed" in pytorch
        assert "cuda_available" in pytorch
        assert isinstance(pytorch["cuda_available"], bool)
        assert isinstance(pytorch["devices"], list)

    def test_nvidia_smi_function(self):
        smi = check_nvidia_smi()
        assert isinstance(smi, dict)
        assert "available" in smi

