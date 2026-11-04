"""
CrowdPulse - Database Models (Phase 3)
Project: CrowdPulse - Crowd Density Management System (SIH25165)
Phase: Phase 3 Spatial Analytics & Incident Database

Beginner-Friendly Explanation:
This file defines the database structure (schema) using SQLAlchemy ORM.
ORM (Object Relational Mapping) lets us interact with database tables as normal
Python classes and objects instead of writing raw SQL queries.

We store:
1. Zone: Information about each physical area (name, safe capacity, camera)
2. DensityLog: Periodic records of crowd counts and occupancy percentages
3. IncidentAlert: Warnings logged whenever overcrowding or bottlenecks occur
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def utc_now():
    """Returns current UTC timestamp without timezone offset for SQLite compatibility."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Zone(Base):
    """Represents a physical monitored area (e.g., North Gate, Central Plaza)."""

    __tablename__ = "zones"

    id = Column(String(50), primary_key=True)  # e.g. "zone_1"
    name = Column(String(100), nullable=False)
    camera_source = Column(String(100), nullable=False)
    capacity_limit = Column(Integer, nullable=False, default=100)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utc_now)

    # Relationships
    density_logs = relationship("DensityLog", back_populates="zone", cascade="all, delete-orphan")
    alerts = relationship("IncidentAlert", back_populates="zone", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Zone(id='{self.id}', name='{self.name}', capacity={self.capacity_limit})>"


class DensityLog(Base):
    """Stores timestamped crowd readings for temporal analysis and auditing."""

    __tablename__ = "density_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    zone_id = Column(String(50), ForeignKey("zones.id"), nullable=False, index=True)
    timestamp = Column(DateTime, default=utc_now, index=True)
    person_count = Column(Integer, nullable=False)
    occupancy_pct = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)  # "NORMAL", "MODERATE", "CRITICAL"
    bottleneck_detected = Column(Boolean, default=False)
    inference_time_ms = Column(Float, nullable=True)

    # Relationship back to Zone
    zone = relationship("Zone", back_populates="density_logs")

    def __repr__(self):
        return f"<DensityLog(zone='{self.zone_id}', count={self.person_count}, risk='{self.risk_level}')>"


class IncidentAlert(Base):
    """Records high-risk incidents, bottlenecks, or safety violations."""

    __tablename__ = "incident_alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    zone_id = Column(String(50), ForeignKey("zones.id"), nullable=False, index=True)
    timestamp = Column(DateTime, default=utc_now, index=True)
    alert_type = Column(String(50), nullable=False)  # "OVERCROWDING", "BOTTLENECK", "RAPID_SURGE"
    severity = Column(String(20), nullable=False)    # "MEDIUM", "HIGH", "CRITICAL"
    person_count = Column(Integer, nullable=False)
    capacity_limit = Column(Integer, nullable=False)
    description = Column(Text, nullable=False)
    status = Column(String(20), default="ACTIVE")   # "ACTIVE", "ACKNOWLEDGED", "RESOLVED"
    resolved_at = Column(DateTime, nullable=True)

    # Relationship back to Zone
    zone = relationship("Zone", back_populates="alerts")

    def __repr__(self):
        return f"<IncidentAlert(id={self.id}, zone='{self.zone_id}', severity='{self.severity}')>"
