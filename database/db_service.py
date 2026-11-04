"""
CrowdPulse - Database Access Service (Phase 3)
Project: CrowdPulse - Crowd Density Management System (SIH25165)
Phase: Phase 3 Spatial Analytics & Incident Database

Beginner-Friendly Explanation:
This module handles all database operations (storing readings, querying history,
and logging safety alerts).

We use SQLite, a fast, file-based database that requires zero external server setup.
All data is saved locally in: database/crowdpulse.db.
"""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import random
from typing import Dict, List, Optional
import pandas as pd
from sqlalchemy import create_engine, desc, func
from sqlalchemy.orm import Session, sessionmaker

import config
from database.models import Base, DensityLog, IncidentAlert, Zone, utc_now

# SQLite database file path
DB_FILE = config.DATABASE_DIR / "crowdpulse.db"
DATABASE_URL = f"sqlite:///{DB_FILE}"

# Engine & Session Factory
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Required for SQLite in multi-threaded Streamlit
    echo=False
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db_session():
    """Context manager providing a transactional database session."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db():
    """Creates database tables and seeds initial zones if empty."""
    Base.metadata.create_all(bind=engine)

    with get_db_session() as session:
        existing_zones_count = session.query(Zone).count()
        if existing_zones_count == 0:
            for z in config.DEFAULT_ZONES:
                new_zone = Zone(
                    id=z["id"],
                    name=z["name"],
                    camera_source=z["camera_source"],
                    capacity_limit=z["capacity_limit"],
                    is_active=True,
                )
                session.add(new_zone)
            session.commit()


def log_density_reading(
    zone_id: str,
    person_count: int,
    capacity_limit: Optional[int] = None,
    bottleneck_detected: bool = False,
    inference_time_ms: Optional[float] = None,
    timestamp: Optional[datetime] = None,
) -> Dict:
    """Logs a live crowd reading and triggers an alert if thresholds are breached."""
    if timestamp is None:
        timestamp = utc_now()

    with get_db_session() as session:
        # Fetch zone details if capacity not explicitly given
        zone = session.query(Zone).filter(Zone.id == zone_id).first()
        if zone and capacity_limit is None:
            capacity_limit = zone.capacity_limit
        elif capacity_limit is None:
            capacity_limit = 100

        occupancy_pct = round((person_count / max(capacity_limit, 1)) * 100.0, 1)

        # Classify risk level
        if occupancy_pct < config.DENSITY_THRESHOLDS["NORMAL"]:
            risk_level = "NORMAL"
        elif occupancy_pct < config.DENSITY_THRESHOLDS["MODERATE"]:
            risk_level = "MODERATE"
        else:
            risk_level = "CRITICAL"

        # 1. Insert density log
        log_entry = DensityLog(
            zone_id=zone_id,
            timestamp=timestamp,
            person_count=person_count,
            occupancy_pct=occupancy_pct,
            risk_level=risk_level,
            bottleneck_detected=bottleneck_detected,
            inference_time_ms=inference_time_ms,
        )
        session.add(log_entry)

        # 2. Check if an Incident Alert should be generated
        alert_created = False
        if risk_level == "CRITICAL":
            alert = IncidentAlert(
                zone_id=zone_id,
                timestamp=timestamp,
                alert_type="OVERCROWDING",
                severity="HIGH" if occupancy_pct < 120 else "CRITICAL",
                person_count=person_count,
                capacity_limit=capacity_limit,
                description=f"Critical crowd density ({occupancy_pct}%) detected at {zone.name if zone else zone_id} ({person_count}/{capacity_limit} persons).",
                status="ACTIVE",
            )
            session.add(alert)
            alert_created = True

        elif bottleneck_detected:
            alert = IncidentAlert(
                zone_id=zone_id,
                timestamp=timestamp,
                alert_type="BOTTLENECK_FORMING",
                severity="MEDIUM",
                person_count=person_count,
                capacity_limit=capacity_limit,
                description=f"Spatial bottleneck cluster detected at {zone.name if zone else zone_id}. Crowd flow restricted.",
                status="ACTIVE",
            )
            session.add(alert)
            alert_created = True

        session.commit()

        return {
            "zone_id": zone_id,
            "person_count": person_count,
            "occupancy_pct": occupancy_pct,
            "risk_level": risk_level,
            "bottleneck_detected": bottleneck_detected,
            "alert_created": alert_created,
            "timestamp": timestamp.isoformat(),
        }


def get_recent_logs_dataframe(zone_id: Optional[str] = None, limit: int = 100) -> pd.DataFrame:
    """Fetches recent density logs as a Pandas DataFrame for charting."""
    with get_db_session() as session:
        query = session.query(
            DensityLog.timestamp,
            DensityLog.zone_id,
            Zone.name.label("zone_name"),
            DensityLog.person_count,
            Zone.capacity_limit,
            DensityLog.occupancy_pct,
            DensityLog.risk_level,
            DensityLog.bottleneck_detected,
            DensityLog.inference_time_ms,
        ).join(Zone, DensityLog.zone_id == Zone.id)

        if zone_id and zone_id != "All Zones":
            query = query.filter(DensityLog.zone_id == zone_id)

        query = query.order_by(desc(DensityLog.timestamp)).limit(limit)
        results = query.all()

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame([
            {
                "Timestamp": r.timestamp,
                "Zone ID": r.zone_id,
                "Zone Name": r.zone_name,
                "Count": r.person_count,
                "Capacity": r.capacity_limit,
                "Occupancy (%)": r.occupancy_pct,
                "Risk": r.risk_level,
                "Bottleneck": "⚠️ Yes" if r.bottleneck_detected else "No",
                "Latency (ms)": r.inference_time_ms,
            }
            for r in reversed(results)
        ])
        return df


def get_active_alerts() -> List[Dict]:
    """Retrieves all active, unresolved alerts."""
    with get_db_session() as session:
        alerts = (
            session.query(IncidentAlert, Zone.name)
            .join(Zone, IncidentAlert.zone_id == Zone.id)
            .filter(IncidentAlert.status != "RESOLVED")
            .order_by(desc(IncidentAlert.timestamp))
            .all()
        )

        return [
            {
                "id": a.IncidentAlert.id,
                "zone_id": a.IncidentAlert.zone_id,
                "zone_name": a.name,
                "timestamp": a.IncidentAlert.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "alert_type": a.IncidentAlert.alert_type,
                "severity": a.IncidentAlert.severity,
                "person_count": a.IncidentAlert.person_count,
                "capacity_limit": a.IncidentAlert.capacity_limit,
                "description": a.IncidentAlert.description,
                "status": a.IncidentAlert.status,
            }
            for a in alerts
        ]


def update_alert_status(alert_id: int, new_status: str):
    """Updates the status of an alert (e.g. 'ACKNOWLEDGED' or 'RESOLVED')."""
    with get_db_session() as session:
        alert = session.query(IncidentAlert).filter(IncidentAlert.id == alert_id).first()
        if alert:
            alert.status = new_status
            if new_status == "RESOLVED":
                alert.resolved_at = utc_now()
            session.commit()


def seed_mock_history_if_empty(num_hours: int = 4):
    """Seeds realistic historical time-series data if the logs table is empty

    so graphs display immediately upon launching the app.
    """
    init_db()

    with get_db_session() as session:
        if session.query(DensityLog).count() > 0:
            return  # Already has logs

        now = utc_now()
        for hour_offset in range(num_hours * 12, 0, -1):  # Log every 5 minutes
            t = now - timedelta(minutes=hour_offset * 5)
            # Create readings for each of the 4 zones
            for zone in session.query(Zone).all():
                # Base counts with periodic wave pattern
                base_pct = random.uniform(0.3, 0.85)
                # Zone 3 often gets crowded
                if zone.id == "zone_3":
                    base_pct = min(1.15, base_pct + 0.25)

                count = int(zone.capacity_limit * base_pct)
                occ_pct = round((count / zone.capacity_limit) * 100.0, 1)

                risk = "NORMAL"
                if occ_pct >= config.DENSITY_THRESHOLDS["MODERATE"]:
                    risk = "CRITICAL"
                elif occ_pct >= config.DENSITY_THRESHOLDS["NORMAL"]:
                    risk = "MODERATE"

                bottleneck = True if (risk == "CRITICAL" and random.random() > 0.4) else False

                log = DensityLog(
                    zone_id=zone.id,
                    timestamp=t,
                    person_count=count,
                    occupancy_pct=occ_pct,
                    risk_level=risk,
                    bottleneck_detected=bottleneck,
                    inference_time_ms=round(random.uniform(28.0, 48.0), 1),
                )
                session.add(log)

                if risk == "CRITICAL" and random.random() > 0.7:
                    alert = IncidentAlert(
                        zone_id=zone.id,
                        timestamp=t,
                        alert_type="OVERCROWDING",
                        severity="HIGH",
                        person_count=count,
                        capacity_limit=zone.capacity_limit,
                        description=f"Automated Alert: {zone.name} exceeded safe density limit ({occ_pct}% occupancy).",
                        status="RESOLVED" if hour_offset > 6 else "ACTIVE",
                        resolved_at=t + timedelta(minutes=15) if hour_offset > 6 else None,
                    )
                    session.add(alert)

        session.commit()
