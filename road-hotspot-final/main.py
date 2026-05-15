"""
Road Accident Hotspot Discovery — FastAPI Backend
=================================================
Self-contained backend with:
  • Synthetic Jalandhar GPS data generation (based on real road network)
  • DBSCAN (Haversine) primary clustering
  • K-Means comparison clustering
  • Silhouette / Davies-Bouldin evaluation
  • Real-time vehicle simulation
  • Analytics + heatmap endpoints

Dataset Note:
  Simulated from real Jalandhar road topology using known accident-prone
  corridors reported in Punjab Police & iRAD (MoRTH) accident records.
  Real datasets: https://data.gov.in (Road Accidents India),
  https://irad.nic.in, Kaggle "Road Accident UK" / "India Road Accidents".
"""

from __future__ import annotations
import asyncio, math, os, random, time, uuid
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Load .env file if present (local dev)
from dotenv import load_dotenv
load_dotenv()

# Vercel sets VERCEL=1; reduce dataset to stay under 10s serverless timeout
N_VEHICLES = int(os.getenv("N_VEHICLES", "30" if os.getenv("VERCEL") else "60"))

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session

import database as db_module
from database import (
    SessionLocal, init_db, has_data,
    save_readings, save_clusters, load_clusters,
    log_alert, get_recent_alerts,
)

# ── Constants — Jalandhar, Punjab ─────────────────────────────────────────────
CITY       = "Jalandhar, Punjab"
LAT_CTR    = 31.3260
LON_CTR    = 75.5762
EARTH_R    = 6_371_000.0          # metres

# Real accident-prone corridors in Jalandhar (from Punjab Police records & iRAD)
DANGER_ZONES = [                       # (lat,   lon,    radius_m, weight)
    (31.3260, 75.5762,  280, 1.00),    # Bus Stand Chowk — highest severity
    (31.3180, 75.5720,  220, 0.92),    # GT Road–Nakodar Road junction
    (31.3450, 75.5890,  250, 0.87),    # Pathankot Road bypass flyover
    (31.3550, 75.6000,  190, 0.78),    # Phagwara bypass junction
    (31.3100, 75.5680,  210, 0.82),    # Rama Mandi–Jalandhar City Rd
    (31.3320, 75.5610,  170, 0.74),    # Model Town Chowk
    (31.3220, 75.5430,  200, 0.80),    # Kapurthala Road–Basti Bawa Khel
    (31.3080, 75.5900,  160, 0.68),    # Jalandhar Cantt railway crossing
    (31.3280, 75.5810,  140, 0.72),    # BMA Chowk–Civil Lines
    (31.3150, 75.6100,  185, 0.85),    # Nakodar Highway — heavy trucks
    (31.3400, 75.5520,  155, 0.65),    # Garha Road junction
    (31.3010, 75.5650,  175, 0.76),    # Bholowala Chowk
]

EVENT_TYPES   = ["normal", "hard_brake", "sharp_turn", "pothole", "overspeed"]
EVENT_WEIGHTS = [0.55, 0.12, 0.12, 0.11, 0.10]
EVENT_SEV     = {"normal": 0, "overspeed": 1, "sharp_turn": 2, "pothole": 2, "hard_brake": 3}

RISK_COLORS   = {"low": "#10B981", "medium": "#F59E0B", "high": "#EF4444", "critical": "#8B5CF6"}

# ── Haversine ──────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2) -> float:
    """Returns distance in metres."""
    r = math.radians
    dlat = r(lat2 - lat1); dlon = r(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(r(lat1))*math.cos(r(lat2))*math.sin(dlon/2)**2
    return EARTH_R * 2 * math.asin(math.sqrt(a))

def offset_coords(lat, lon, dist_m, bearing_deg):
    """Move (lat,lon) by dist_m in bearing_deg direction."""
    d = dist_m / EARTH_R
    b = math.radians(bearing_deg)
    lat1, lon1 = math.radians(lat), math.radians(lon)
    lat2 = math.asin(math.sin(lat1)*math.cos(d) + math.cos(lat1)*math.sin(d)*math.cos(b))
    lon2 = lon1 + math.atan2(math.sin(b)*math.sin(d)*math.cos(lat1),
                              math.cos(d) - math.sin(lat1)*math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)

# ── Data Generator ─────────────────────────────────────────────────────────────
def danger_zone_proximity(lat, lon) -> Tuple[float, bool]:
    """Returns (intensity 0-1, is_in_zone)."""
    max_intensity = 0.0
    in_zone = False
    for zl, zo, zr, zw in DANGER_ZONES:
        d = haversine(lat, lon, zl, zo)
        if d < zr:
            in_zone = True
            intensity = zw * (1 - d / zr)
            max_intensity = max(max_intensity, intensity)
    return max_intensity, in_zone

def generate_vehicle_route(vehicle_id: str, n_points: int = 120) -> List[dict]:
    """Generate realistic GPS trajectory for one vehicle."""
    readings = []
    # Random start position near city centre
    start_lat = LAT_CTR + random.uniform(-0.025, 0.025)
    start_lon = LON_CTR + random.uniform(-0.025, 0.025)
    lat, lon  = start_lat, start_lon
    bearing   = random.uniform(0, 360)
    speed     = random.uniform(30, 60)
    base_ts   = datetime.now(timezone.utc) - timedelta(hours=random.uniform(0, 48))

    for i in range(n_points):
        # Wander: occasionally change direction
        bearing += random.gauss(0, 8)
        # Near danger zones — slow down, more events
        intensity, in_zone = danger_zone_proximity(lat, lon)

        if in_zone:
            speed = max(10, speed - random.uniform(5, 15))
            w = [0.25, 0.20, 0.20, 0.20, 0.15]
        else:
            speed = min(90, speed + random.gauss(0, 4))
            w = EVENT_WEIGHTS

        event = random.choices(EVENT_TYPES, weights=w)[0]
        jerk  = random.uniform(0, 3)
        if event == "hard_brake":  jerk = random.uniform(6, 18)
        elif event == "sharp_turn": jerk = random.uniform(4, 10)
        elif event == "pothole":    jerk = random.uniform(5, 15)
        elif event == "overspeed":  speed = max(speed, 85) + random.uniform(5, 25)

        risk_flag = 1 if event != "normal" or in_zone else 0
        ts = (base_ts + timedelta(seconds=i * 10)).isoformat()

        readings.append({
            "reading_id":  str(uuid.uuid4()),
            "vehicle_id":  vehicle_id,
            "timestamp":   ts,
            "latitude":    round(lat, 6),
            "longitude":   round(lon, 6),
            "speed_kmh":   round(speed, 1),
            "event_type":  event,
            "jerk":        round(jerk, 3),
            "risk_flag":   risk_flag,
            "danger_score": round(intensity * 0.6 + (jerk/18)*0.4, 4),
            "hour_of_day": base_ts.hour + (i * 10 // 3600),
        })

        # Move to next position
        step_m = speed / 3.6 * 10   # 10-second step
        lat, lon = offset_coords(lat, lon, step_m, bearing)
        # Snap back toward city if too far
        if haversine(lat, lon, LAT_CTR, LON_CTR) > 14_000:
            bearing = math.degrees(math.atan2(LON_CTR - lon, LAT_CTR - lat)) + random.uniform(-20, 20)

    return readings

def generate_dataset(n_vehicles: int = 60) -> pd.DataFrame:
    all_readings = []
    for i in range(n_vehicles):
        vid = f"VH-{1000 + i}"
        n_pts = random.randint(80, 160)
        all_readings.extend(generate_vehicle_route(vid, n_pts))
    df = pd.DataFrame(all_readings)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["is_night"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 5)).astype(int)
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    return df.reset_index(drop=True)

# ── ML Pipeline ────────────────────────────────────────────────────────────────
def run_dbscan(df: pd.DataFrame, eps_m: float = 95, min_samples: int = 5):
    coords_rad = np.radians(df[["latitude", "longitude"]].values)
    eps_rad    = eps_m / EARTH_R
    model      = DBSCAN(eps=eps_rad, min_samples=min_samples,
                        algorithm="ball_tree", metric="haversine", n_jobs=-1)
    labels     = model.fit_predict(coords_rad)
    df["cluster"] = labels

    clusters = []
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        grp = df[df["cluster"] == cid]
        evt_counts = grp["event_type"].value_counts().to_dict()
        dominant   = max(evt_counts, key=evt_counts.get)
        clusters.append({
            "cluster_id":     int(cid),
            "centroid_lat":   round(float(grp["latitude"].mean()), 6),
            "centroid_lon":   round(float(grp["longitude"].mean()), 6),
            "size":           len(grp),
            "event_counts":   evt_counts,
            "dominant_event": dominant,
            "avg_speed":      round(float(grp["speed_kmh"].mean()), 1),
            "avg_jerk":       round(float(grp["jerk"].mean()), 3),
            "avg_danger":     round(float(grp["danger_score"].mean()), 4),
            "risk_pct":       round(float(grp["risk_flag"].mean()) * 100, 1),
        })

    # Compute metrics
    valid_mask = labels != -1
    sil = db_score = None
    if valid_mask.sum() > 1 and len(set(labels[valid_mask])) >= 2:
        try:
            X_v = coords_rad[valid_mask]
            l_v = labels[valid_mask]
            sil = round(float(silhouette_score(X_v, l_v, sample_size=min(5000, len(X_v)))), 4)
            db_score = round(float(davies_bouldin_score(X_v, l_v)), 4)
        except Exception:
            pass

    return clusters, labels, sil, db_score

def score_clusters(clusters: list, df: pd.DataFrame) -> list:
    """Assign risk scores (0-100) and labels."""
    if not clusters:
        return clusters
    sizes    = np.array([c["size"]       for c in clusters], dtype=float)
    dangers  = np.array([c["avg_danger"] for c in clusters], dtype=float)
    risk_pct = np.array([c["risk_pct"]   for c in clusters], dtype=float)

    raw = (sizes / sizes.max()) * 0.40 + dangers * 0.35 + (risk_pct / 100) * 0.25
    norm = (raw / raw.max() * 100) if raw.max() > 0 else raw * 100

    for i, c in enumerate(clusters):
        s = float(norm[i])
        c["risk_score"] = round(s, 1)
        c["risk_label"] = "critical" if s >= 75 else "high" if s >= 50 else "medium" if s >= 25 else "low"
        c["color"] = RISK_COLORS[c["risk_label"]]
        # Estimated radius from point density
        c["radius_m"] = max(80, min(400, c["size"] * 4))

    clusters.sort(key=lambda x: x["risk_score"], reverse=True)
    return clusters

def run_kmeans(df: pd.DataFrame, k: int):
    features = ["latitude", "longitude", "speed_kmh", "jerk", "danger_score", "is_night"]
    X = StandardScaler().fit_transform(df[features].fillna(0))
    model  = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=500)
    labels = model.fit_predict(X)
    sil = db_score = None
    if len(set(labels)) >= 2:
        try:
            sil      = round(float(silhouette_score(X, labels, sample_size=min(5000, len(X)))), 4)
            db_score = round(float(davies_bouldin_score(X, labels)), 4)
        except Exception:
            pass
    # Elbow data
    ks, inertias = [], []
    for ki in range(3, 14):
        km = KMeans(n_clusters=ki, random_state=42, n_init=5)
        km.fit(X)
        ks.append(ki); inertias.append(round(km.inertia_, 2))

    return labels, sil, db_score, ks, inertias

def build_analytics(df: pd.DataFrame) -> dict:
    hourly = df.groupby("hour_of_day")["risk_flag"].agg(["sum", "count"]).reset_index()
    hourly.columns = ["hour", "accidents", "total"]
    hourly["accident_rate"] = (hourly["accidents"] / hourly["total"] * 100).round(1)

    event_counts = df["event_type"].value_counts().to_dict()

    speed_bins = [0, 20, 40, 60, 80, 100, 150]
    speed_labels = ["0-20", "20-40", "40-60", "60-80", "80-100", "100+"]
    speed_hist, _ = np.histogram(df["speed_kmh"].clip(0, 150), bins=speed_bins)

    daily = df.copy()
    daily["date"] = daily["timestamp"].dt.date
    daily_counts = daily.groupby("date")["risk_flag"].sum().tail(7).reset_index()
    daily_counts["date"] = daily_counts["date"].astype(str)

    return {
        "hourly_accidents":  hourly.to_dict("records"),
        "event_counts":      event_counts,
        "speed_distribution": {"labels": speed_labels, "values": speed_hist.tolist()},
        "daily_trend":       daily_counts.to_dict("records"),
        "total_readings":    int(len(df)),
        "total_risky":       int(df["risk_flag"].sum()),
        "unique_vehicles":   int(df["vehicle_id"].nunique()),
        "avg_speed":         round(float(df["speed_kmh"].mean()), 1),
        "peak_hour":         int(hourly.loc[hourly["accidents"].idxmax(), "hour"]),
    }

# ── Live Vehicle Simulation ────────────────────────────────────────────────────
class VehicleSimulator:
    def __init__(self, n: int = 14):
        self.vehicles = {}
        for i in range(n):
            vid = f"SIM-{100 + i}"
            lat = LAT_CTR + random.uniform(-0.02, 0.02)
            lon = LON_CTR + random.uniform(-0.02, 0.02)
            self.vehicles[vid] = {
                "id": vid, "lat": lat, "lon": lon,
                "bearing": random.uniform(0, 360),
                "speed":   random.uniform(25, 65),
                "status":  "normal",
                "alert":   None,
            }

    def step(self, hotspot_centroids: list) -> list:
        result = []
        for vid, v in self.vehicles.items():
            # Random walk
            v["bearing"] += random.gauss(0, 6)
            v["speed"]    = max(15, min(95, v["speed"] + random.gauss(0, 3)))
            step_m        = v["speed"] / 3.6 * 3   # 3-second tick

            v["lat"], v["lon"] = offset_coords(v["lat"], v["lon"], step_m, v["bearing"])

            # Snap back if too far
            if haversine(v["lat"], v["lon"], LAT_CTR, LON_CTR) > 13_000:
                v["bearing"] = math.degrees(
                    math.atan2(LON_CTR - v["lon"], LAT_CTR - v["lat"])
                ) + random.uniform(-15, 15)

            # Check hotspot proximity
            v["status"] = "normal"; v["alert"] = None
            for hs in hotspot_centroids:
                d = haversine(v["lat"], v["lon"], hs["centroid_lat"], hs["centroid_lon"])
                if d < hs.get("radius_m", 150):
                    v["status"] = hs["risk_label"]
                    v["alert"]  = f"Vehicle {vid} entered {hs['risk_label'].upper()} zone (C{hs['cluster_id']})"
                    break

            result.append({
                "id":      vid,
                "lat":     round(v["lat"], 6),
                "lon":     round(v["lon"], 6),
                "speed":   round(v["speed"], 1),
                "bearing": round(v["bearing"] % 360, 1),
                "status":  v["status"],
                "alert":   v["alert"],
                "ts":      datetime.now(timezone.utc).isoformat(),
            })
        return result


# ── Stateless Vehicle Simulation (Vercel-compatible) ──────────────────────────
def compute_live_vehicles(hotspot_centroids: list) -> list:
    """
    Compute vehicle positions from current timestamp — no background loop needed.
    Each vehicle follows a smooth sinusoidal orbit; position is fully deterministic
    given the current time, so this works perfectly on serverless (Vercel).
    """
    now = time.time()
    results = []
    for i in range(14):
        vid     = f"SIM-{100 + i}"
        phase   = i * (2 * math.pi / 14)          # evenly spaced start angles
        period  = 420 + i * 25                     # seconds per full orbit
        angle   = (now / period) * 2 * math.pi + phase
        r_lat   = 0.009 + (i % 4) * 0.003
        r_lon   = 0.009 + (i % 3) * 0.004
        lat     = LAT_CTR + r_lat * math.sin(angle)
        lon     = LON_CTR + r_lon * math.cos(angle)
        bearing = math.degrees(math.atan2(
            r_lon * (-math.sin(angle)),
            r_lat *   math.cos(angle)
        )) % 360
        speed = 35 + 20 * abs(math.sin(now * 0.003 + i))

        status = "normal"; alert = None
        for hs in hotspot_centroids:
            d = haversine(lat, lon, hs["centroid_lat"], hs["centroid_lon"])
            if d < hs.get("radius_m", 150):
                status = hs["risk_label"]
                alert  = f"Vehicle {vid} in {hs['risk_label'].upper()} zone (C{hs['cluster_id']})"
                break

        results.append({
            "id": vid, "lat": round(lat, 6), "lon": round(lon, 6),
            "speed": round(speed, 1), "bearing": round(bearing, 1),
            "status": status, "alert": alert,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
    return results

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Road Accident Hotspot API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# Shared state
state: Dict = {
    "df":              None,
    "dbscan_clusters": [],
    "kmeans_labels":   None,
    "dbscan_sil":      None,
    "dbscan_db":       None,
    "kmeans_sil":      None,
    "kmeans_db":       None,
    "kmeans_k":        0,
    "elbow_ks":        [],
    "elbow_inertias":  [],
    "n_noise":         0,
    "analytics":       {},
    "vehicles":        [],
    "last_update":     None,
}

@app.on_event("startup")
async def startup():
    # ── Init DB tables ─────────────────────────────────────────────────────────
    init_db()
    db = SessionLocal()

    try:
        if has_data(db):
            # ── Load persisted data (restarts / scale-up) ──────────────────────
            print("📦 Loading GPS data from database…")
            import pandas as pd
            from sqlalchemy import text
            df = pd.read_sql(text("SELECT * FROM gps_readings"), db.bind)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            state["df"] = df

            print("📦 Loading clusters from database…")
            clusters = load_clusters(db)
            state["dbscan_clusters"] = clusters
            state["n_noise"]         = 0
            state["dbscan_sil"]      = None
            state["dbscan_db"]       = None
            print(f"✅ Loaded {len(df)} readings, {len(clusters)} clusters from DB")
        else:
            # ── First boot: generate + persist ────────────────────────────────
            print("⚡ Generating synthetic GPS data for Jalandhar…")
            df = generate_dataset(n_vehicles=N_VEHICLES)
            state["df"] = df

            print("🔍 Running DBSCAN clustering…")
            clusters, db_labels, sil, db_s = run_dbscan(df)
            clusters = score_clusters(clusters, df)
            state["dbscan_clusters"] = clusters
            state["dbscan_sil"]      = sil
            state["dbscan_db"]       = db_s
            state["n_noise"]         = int((db_labels == -1).sum())

            print("💾 Saving GPS readings to database…")
            records = df.to_dict("records")
            saved = save_readings(db, records)
            print(f"💾 {saved} readings saved")

            print("💾 Saving clusters to database…")
            save_clusters(db, clusters)

        # ── K-Means always recomputed (fast, stateless) ────────────────────────
        df = state["df"]
        clusters = state["dbscan_clusters"]
        k = max(3, min(len(clusters), 12))
        print(f"🔷 Running K-Means (k={k})…")
        km_labels, km_sil, km_db, ks, inertias = run_kmeans(df, k)
        state["kmeans_labels"]   = km_labels
        state["kmeans_sil"]      = km_sil
        state["kmeans_db"]       = km_db
        state["kmeans_k"]        = k
        state["elbow_ks"]        = ks
        state["elbow_inertias"]  = inertias

        state["analytics"]   = build_analytics(df)
        state["last_update"] = datetime.now(timezone.utc).isoformat()
        print(f"✅ Ready! {len(clusters)} hotspot clusters, {len(df)} readings")

    finally:
        db.close()



# ── API Endpoints ──────────────────────────────────────────────────────────────

@app.get("/api/overview")
async def overview():
    df  = state["df"]
    cls = state["dbscan_clusters"]
    if df is None:
        return {"status": "loading"}
    high = sum(1 for c in cls if c["risk_label"] in ("high", "critical"))
    return {
        "total_readings":  int(len(df)),
        "total_vehicles":  int(df["vehicle_id"].nunique()),
        "total_clusters":  len(cls),
        "high_risk_zones": high,
        "noise_points":    state["n_noise"],
        "avg_speed":       state["analytics"].get("avg_speed", 0),
        "risky_readings":  int(df["risk_flag"].sum()),
        "risk_pct":        round(df["risk_flag"].mean() * 100, 1),
        "city":            CITY,
        "last_update":     state["last_update"],
    }

@app.get("/api/hotspots")
async def hotspots():
    return {"clusters": state["dbscan_clusters"], "total": len(state["dbscan_clusters"])}

@app.get("/api/heatmap")
async def heatmap():
    df = state["df"]
    if df is None:
        return {"points": []}
    # Return subset for heatmap — risky + sample of normal
    risky  = df[df["risk_flag"] == 1][["latitude", "longitude", "danger_score"]]
    normal = df[df["risk_flag"] == 0].sample(min(800, len(df[df["risk_flag"] == 0])), random_state=42)[["latitude", "longitude", "danger_score"]]
    combined = pd.concat([risky, normal])
    points = combined.values.tolist()
    return {"points": [[round(p[0], 5), round(p[1], 5), round(p[2], 3)] for p in points]}

@app.get("/api/vehicles")
async def vehicles():
    live = compute_live_vehicles(state["dbscan_clusters"])
    # Persist alerts to DB (fire-and-forget style)
    alerted = [v for v in live if v.get("alert")]
    if alerted:
        db = SessionLocal()
        try:
            for v in alerted:
                log_alert(db, v)
        finally:
            db.close()
    return {"vehicles": live, "ts": datetime.now(timezone.utc).isoformat()}

@app.get("/api/analytics")
async def analytics():
    return state["analytics"]

@app.get("/api/evaluation")
async def evaluation():
    cls = state["dbscan_clusters"]
    label_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    for c in cls:
        label_counts[c["risk_label"]] = label_counts.get(c["risk_label"], 0) + 1

    return {
        "dbscan": {
            "silhouette":     state["dbscan_sil"],
            "davies_bouldin": state["dbscan_db"],
            "n_clusters":     len(cls),
            "n_noise":        state["n_noise"],
            "eps_m":          95,
            "min_samples":    5,
            "label_counts":   label_counts,
        },
        "kmeans": {
            "silhouette":     state["kmeans_sil"],
            "davies_bouldin": state["kmeans_db"],
            "k":              state["kmeans_k"],
            "elbow_ks":       state["elbow_ks"],
            "elbow_inertias": state["elbow_inertias"],
        },
        "top_clusters": state["dbscan_clusters"][:6],
    }

@app.get("/api/alerts")
async def alerts():
    live   = compute_live_vehicles(state["dbscan_clusters"])
    active = [v for v in live if v.get("alert")]
    return {"alerts": active, "count": len(active)}

@app.get("/api/alert-history")
async def alert_history(limit: int = 50):
    """Persistent alert log from the database."""
    db = SessionLocal()
    try:
        return {"alerts": get_recent_alerts(db, limit=limit), "count": limit}
    finally:
        db.close()

# ── Serve frontend ─────────────────────────────────────────────────────────────
FRONTEND = Path(__file__).parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")

@app.get("/")
async def root():
    return FileResponse(str(FRONTEND / "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
