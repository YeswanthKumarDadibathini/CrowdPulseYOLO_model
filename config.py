"""
CrowdPulse - Configuration Module
Project: CrowdPulse - Crowd Density Management System (SIH25165)
Phase: Phase 1 Foundation

This file contains global settings, path configurations, and default zone definitions.
Designed to be beginner-friendly, clean, and easily editable.
"""

from pathlib import Path

# ==========================================
# 1. Project Information
# ==========================================
PROJECT_NAME = "CrowdPulse"
SIH_PROJECT_CODE = "SIH25165"
PROJECT_TAGLINE = "Smart Crowd Density Monitoring & Early Incident Prevention"
APP_VERSION = "0.1.0-phase1"
AUTHOR = "B.Tech Final Year Team"

# ==========================================
# 2. Directory Paths
# ==========================================
# Path(__file__).resolve().parent gets the root folder 'CrowdPulse'
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
VIDEOS_DIR = DATA_DIR / "videos"
DATASETS_DIR = DATA_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"
DATABASE_DIR = BASE_DIR / "database"

# Ensure runtime directories exist
for path in [DATA_DIR, VIDEOS_DIR, DATASETS_DIR, MODELS_DIR, DATABASE_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# ==========================================
# 3. Density Thresholds (Percentages)
# ==========================================
# Density is calculated as (current_count / capacity_limit) * 100
DENSITY_THRESHOLDS = {
    "NORMAL": 60.0,       # Below 60%: Safe (Green)
    "MODERATE": 80.0,     # 60% to 80%: Caution (Yellow)
    "CRITICAL": 100.0,    # Above 80%: Overcrowded / Alert (Red)
}

# ==========================================
# 4. Default Monitored Zones (Phase 1 Placeholders)
# ==========================================
# Each zone represents a monitored physical location with a fixed safe capacity.
DEFAULT_ZONES = [
    {
        "id": "zone_1",
        "name": "North Gate (Main Entrance)",
        "camera_source": "Cam 01 - North Turnstiles",
        "capacity_limit": 150,
        "current_count": 45,
        "demo_status": "Normal",
        "risk_level": "Low",
    },
    {
        "id": "zone_2",
        "name": "Central Plaza",
        "camera_source": "Cam 02 - Central Open Area",
        "capacity_limit": 500,
        "current_count": 390,
        "demo_status": "Moderate",
        "risk_level": "Medium",
    },
    {
        "id": "zone_3",
        "name": "Food Concourse & Walkway",
        "camera_source": "Cam 03 - Concourse Corridor",
        "capacity_limit": 250,
        "current_count": 230,
        "demo_status": "High Density",
        "risk_level": "High",
    },
    {
        "id": "zone_4",
        "name": "South Exit & Transit Gate",
        "camera_source": "Cam 04 - South Escalators",
        "capacity_limit": 200,
        "current_count": 60,
        "demo_status": "Normal",
        "risk_level": "Low",
    },
]


def get_zone_by_id(zone_id: str):
    """Retrieve zone settings dictionary given its unique ID."""
    for zone in DEFAULT_ZONES:
        if zone["id"] == zone_id:
            return zone
    return None

