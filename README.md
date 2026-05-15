# 🛣️ TrafficSense — Road Accident Hotspot Intelligence Dashboard

**AI-powered smart city traffic analytics platform** built on unsupervised machine learning (DBSCAN + K-Means) with a premium real-time dashboard.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the server
```bash
python main.py
```

### 3. Open dashboard
Visit → **http://localhost:8000**

---

## 🗺️ Dashboard Pages

| Page | Description |
|------|-------------|
| **Overview** | KPI cards, mini map, risk donut, alert feed, top danger zones |
| **Live Map** | Full Leaflet map — heatmap, cluster circles, live vehicles, legend |
| **Analytics** | Hourly accidents, event breakdown, speed histogram, weekly trend, cluster scorecard |
| **Model Eval** | Silhouette/Davies-Bouldin scores, DBSCAN vs K-Means comparison, elbow curve |
| **Live Tracking** | Real-time vehicle status board, zone alert feed, simulation statistics |
| **Hotspot Zones** | Complete DBSCAN cluster registry with all metrics |

---

## 🤖 ML Pipeline

```
GPS Data Generation (Ludhiana, Punjab)
         ↓
Feature Engineering (lat, lon, speed, jerk, danger_score, is_night)
         ↓
┌──────────────────┐    ┌──────────────────┐
│   DBSCAN         │    │   K-Means        │
│ (Haversine dist) │    │ (scaled features)│
│ eps=95m, min=5   │    │ k = auto-selected│
└────────┬─────────┘    └────────┬─────────┘
         │                       │
    Risk Scoring              Elbow Analysis
    (0-100 score)          Silhouette / DB Index
         │
    [Low / Medium / High / Critical]
```

### Evaluation Metrics
- **Silhouette Score**: cluster cohesion/separation (↑ better, range −1 to 1)
- **Davies-Bouldin Index**: cluster overlap (↓ better, ≥ 0)

---

## 🏙️ Dataset
Synthetic GPS trajectories generated around **Ludhiana, Punjab** (30.9010°N, 75.8573°E) with:
- 60 simulated vehicles
- ~7,000 GPS readings
- 10 pre-seeded danger zones (calibrated from config)
- Event types: `normal`, `hard_brake`, `sharp_turn`, `pothole`, `overspeed`

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Uvicorn |
| ML | scikit-learn (DBSCAN, K-Means, Silhouette, Davies-Bouldin) |
| Frontend | Vanilla HTML/CSS/JS (no build step) |
| Maps | Leaflet.js + leaflet.heat |
| Charts | Chart.js 4 |
| Fonts | Google Fonts (Rajdhani, Exo 2, JetBrains Mono) |

---

## 📡 API Endpoints

```
GET /api/overview      — KPIs, summary stats
GET /api/hotspots      — DBSCAN cluster list with risk scores
GET /api/heatmap       — GPS point cloud for Leaflet.heat
GET /api/vehicles      — Simulated live vehicle positions
GET /api/analytics     — Hourly, event, speed, trend data
GET /api/evaluation    — Silhouette, DB Index, elbow curve
GET /api/alerts        — Vehicles currently in alert zones
```

---

## 🎨 Design System
- **Theme**: Void dark (default) + clean light mode
- **Typography**: Rajdhani (headers) · Exo 2 (body) · JetBrains Mono (data)
- **Colors**: Amber accent · Cyan info · Red danger · Purple critical · Green safe
- **Effects**: Glassmorphism cards · CSS backdrop-filter · Animated KPIs · Heatmap glow

---

*Built for faculty demonstration, placement portfolio, and smart-city research.*
