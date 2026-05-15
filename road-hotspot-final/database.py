"""
database.py — SQLAlchemy models + DB connection
================================================
Supports:
  • SQLite  (default, local dev)  →  DATABASE_URL=sqlite:///./hotspot.db
  • PostgreSQL (Render/production) →  DATABASE_URL=postgresql://user:pass@host/db

Tables
------
  gps_readings      — persistent synthetic GPS dataset
  hotspot_clusters  — DBSCAN computed clusters (risk scores, centroids)
  vehicle_alerts    — live alert log (vehicles entering hotspot zones)
"""

import os, json
from datetime import datetime, timezone
from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    Boolean, DateTime, Text, Index, func
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session

# ── Connection ─────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./hotspot.db")

# Render gives postgres:// URLs; SQLAlchemy 2.x requires postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# ── Base ───────────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass

# ── Models ─────────────────────────────────────────────────────────────────────

class GPSReading(Base):
    """One GPS data point from a vehicle."""
    __tablename__ = "gps_readings"

    id           = Column(Integer, primary_key=True, index=True)
    reading_id   = Column(String(36), unique=True, index=True, nullable=False)
    vehicle_id   = Column(String(20), index=True, nullable=False)
    timestamp    = Column(DateTime(timezone=True), nullable=False)
    latitude     = Column(Float, nullable=False)
    longitude    = Column(Float, nullable=False)
    speed_kmh    = Column(Float)
    event_type   = Column(String(20))
    jerk         = Column(Float)
    risk_flag    = Column(Integer, default=0)
    danger_score = Column(Float, default=0.0)
    hour_of_day  = Column(Integer)
    is_night     = Column(Integer, default=0)
    day_of_week  = Column(Integer)
    cluster_id   = Column(Integer, nullable=True)   # filled after DBSCAN

    __table_args__ = (
        Index("ix_gps_vehicle_ts", "vehicle_id", "timestamp"),
        Index("ix_gps_risk",       "risk_flag"),
    )


class HotspotCluster(Base):
    """DBSCAN cluster (hotspot zone) with risk metrics."""
    __tablename__ = "hotspot_clusters"

    id             = Column(Integer, primary_key=True, index=True)
    cluster_id     = Column(Integer, unique=True, nullable=False)
    centroid_lat   = Column(Float, nullable=False)
    centroid_lon   = Column(Float, nullable=False)
    size           = Column(Integer)
    dominant_event = Column(String(30))
    avg_speed      = Column(Float)
    avg_jerk       = Column(Float)
    avg_danger     = Column(Float)
    risk_pct       = Column(Float)
    risk_score     = Column(Float)
    risk_label     = Column(String(10))
    color          = Column(String(10))
    radius_m       = Column(Integer)
    event_counts   = Column(Text)    # JSON string
    created_at     = Column(DateTime(timezone=True),
                            default=lambda: datetime.now(timezone.utc))


class VehicleAlert(Base):
    """Log of real-time alerts when simulated vehicles enter hotspot zones."""
    __tablename__ = "vehicle_alerts"

    id          = Column(Integer, primary_key=True, index=True)
    vehicle_id  = Column(String(20), index=True, nullable=False)
    alert_msg   = Column(String(255))
    risk_label  = Column(String(10))
    cluster_id  = Column(Integer)
    latitude    = Column(Float)
    longitude   = Column(Float)
    speed       = Column(Float)
    timestamp   = Column(DateTime(timezone=True),
                         default=lambda: datetime.now(timezone.utc),
                         index=True)

    __table_args__ = (
        Index("ix_alert_ts", "timestamp"),
    )


# ── Init ───────────────────────────────────────────────────────────────────────
def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """Yield a DB session (use as dependency or context manager)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Helper functions ──────────────────────────────────────────────────────────

def has_data(db: Session) -> bool:
    """True if GPS readings already exist (skip regeneration on restart)."""
    return db.query(func.count(GPSReading.id)).scalar() > 0


def save_readings(db: Session, records: list[dict]):
    """Bulk-insert GPS readings. Skips duplicates by reading_id."""
    existing = {r[0] for r in db.query(GPSReading.reading_id).all()}
    new_rows = [
        GPSReading(
            reading_id   = r["reading_id"],
            vehicle_id   = r["vehicle_id"],
            timestamp    = r["timestamp"],
            latitude     = r["latitude"],
            longitude    = r["longitude"],
            speed_kmh    = r["speed_kmh"],
            event_type   = r["event_type"],
            jerk         = r["jerk"],
            risk_flag    = r["risk_flag"],
            danger_score = r["danger_score"],
            hour_of_day  = r["hour_of_day"],
            is_night     = r.get("is_night", 0),
            day_of_week  = r.get("day_of_week", 0),
            cluster_id   = r.get("cluster_id"),
        )
        for r in records if r["reading_id"] not in existing
    ]
    if new_rows:
        db.bulk_save_objects(new_rows)
        db.commit()
    return len(new_rows)


def save_clusters(db: Session, clusters: list[dict]):
    """Upsert DBSCAN clusters."""
    db.query(HotspotCluster).delete()
    for c in clusters:
        db.add(HotspotCluster(
            cluster_id     = c["cluster_id"],
            centroid_lat   = c["centroid_lat"],
            centroid_lon   = c["centroid_lon"],
            size           = c["size"],
            dominant_event = c["dominant_event"],
            avg_speed      = c["avg_speed"],
            avg_jerk       = c["avg_jerk"],
            avg_danger     = c["avg_danger"],
            risk_pct       = c["risk_pct"],
            risk_score     = c["risk_score"],
            risk_label     = c["risk_label"],
            color          = c["color"],
            radius_m       = c["radius_m"],
            event_counts   = json.dumps(c.get("event_counts", {})),
        ))
    db.commit()


def load_clusters(db: Session) -> list[dict]:
    """Load all clusters from DB as dicts."""
    rows = db.query(HotspotCluster).order_by(HotspotCluster.risk_score.desc()).all()
    return [
        {
            "cluster_id":     r.cluster_id,
            "centroid_lat":   r.centroid_lat,
            "centroid_lon":   r.centroid_lon,
            "size":           r.size,
            "dominant_event": r.dominant_event,
            "avg_speed":      r.avg_speed,
            "avg_jerk":       r.avg_jerk,
            "avg_danger":     r.avg_danger,
            "risk_pct":       r.risk_pct,
            "risk_score":     r.risk_score,
            "risk_label":     r.risk_label,
            "color":          r.color,
            "radius_m":       r.radius_m,
            "event_counts":   json.loads(r.event_counts or "{}"),
        }
        for r in rows
    ]


def log_alert(db: Session, vehicle: dict):
    """Persist one vehicle alert to the alerts log."""
    db.add(VehicleAlert(
        vehicle_id  = vehicle["id"],
        alert_msg   = vehicle.get("alert"),
        risk_label  = vehicle.get("status"),
        cluster_id  = None,
        latitude    = vehicle.get("lat"),
        longitude   = vehicle.get("lon"),
        speed       = vehicle.get("speed"),
    ))
    db.commit()


def get_recent_alerts(db: Session, limit: int = 50) -> list[dict]:
    """Fetch the most recent alerts from DB."""
    rows = (db.query(VehicleAlert)
              .order_by(VehicleAlert.timestamp.desc())
              .limit(limit)
              .all())
    return [
        {
            "vehicle_id": r.vehicle_id,
            "alert":      r.alert_msg,
            "status":     r.risk_label,
            "lat":        r.latitude,
            "lon":        r.longitude,
            "speed":      r.speed,
            "ts":         r.timestamp.isoformat() if r.timestamp else None,
        }
        for r in rows
    ]
