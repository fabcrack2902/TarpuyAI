#!/usr/bin/env python3
# main.py
import io
import math
import os
import sqlite3
import tempfile
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, List, Dict

import numpy as np
from pathlib import Path
from fastapi import FastAPI, Response, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from fastapi.responses import HTMLResponse
# NDVI display thresholds (ajusta si tu proyecto usa otros rangos)
NDVI_MIN_DISPLAY = -1.0
NDVI_MAX_DISPLAY = 1.0


# --- Configuration
BASE_DIR = Path(__file__).resolve().parent
os.chdir(str(BASE_DIR))  # asegurar ruta base correcta

STATIC_DIR = BASE_DIR / "static"
DB_PATH = os.environ.get("DATA_DB", str(BASE_DIR / "data.db"))
TILE_CACHE_DIR = os.environ.get("TILE_CACHE", str(BASE_DIR / "tiles_cache"))

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ndvi")

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Montar carpeta static
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# --- Rutas principales
@app.get("/")
async def root():
    return FileResponse(BASE_DIR / "index.html")

@app.get("/backup")
async def backup():
    return FileResponse(BASE_DIR / "main_py_backup.html")

# --- Database helpers
def get_db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
    except Exception:
        pass
    return conn


def init_db():
    with get_db_conn() as conn:
        cur = conn.cursor()

        # domain tables
        cur.execute("""CREATE TABLE IF NOT EXISTS stations (id INTEGER PRIMARY KEY, name TEXT, lon REAL, lat REAL)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS areas (id INTEGER PRIMARY KEY, name TEXT, geom TEXT)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS crops (id INTEGER PRIMARY KEY, name TEXT, lon REAL, lat REAL, area REAL)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS pollinators (id INTEGER PRIMARY KEY, name TEXT, lon REAL, lat REAL, abundance INTEGER)""")

        # species metadata and regional mapping
        cur.execute("""
            CREATE TABLE IF NOT EXISTS species_meta (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                common_name TEXT,
                bloom_start_month INTEGER,
                bloom_end_month INTEGER,
                soil TEXT,
                organic_recommendation TEXT,
                notes TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS regional_species (
                id INTEGER PRIMARY KEY,
                region TEXT,
                province TEXT,
                species_name TEXT,
                UNIQUE(region, province, species_name)
            )
        """)

        # seed example data where empty
        cur.execute("SELECT COUNT(*) FROM stations")
        if cur.fetchone()[0] == 0:
            cur.executemany("INSERT INTO stations (name, lon, lat) VALUES (?,?,?)", [
                ("Estacion A", -77.0, -12.0), ("Estacion B", -76.8, -12.2),
                ("Estacion C", -77.03, -12.015), ("Estacion D", -76.985, -12.035),
            ])

        cur.execute("SELECT COUNT(*) FROM areas")
        if cur.fetchone()[0] == 0:
            cur.execute("INSERT INTO areas (name, geom) VALUES (?,?)",
                        ("Area Protegida 1", "POLYGON((-77.05 -12.05, -76.95 -12.05, -76.95 -11.95, -77.05 -11.95, -77.05 -12.05))"))

        cur.execute("SELECT COUNT(*) FROM crops")
        if cur.fetchone()[0] == 0:
            cur.executemany("INSERT INTO crops (name, lon, lat, area) VALUES (?,?,?,?)", [
                ("Maiz", -77.02, -12.01, 12.3), ("Quinoa", -76.99, -12.03, 5.2),
            ])

        cur.execute("SELECT COUNT(*) FROM pollinators")
        if cur.fetchone()[0] == 0:
            cur.executemany("INSERT INTO pollinators (name, lon, lat, abundance) VALUES (?,?,?,?)", [
                ("Abeja A", -77.01, -12.02, 20), ("Abejorro B", -76.98, -12.04, 8),
            ])

        # seed species_meta if empty
        cur.execute("SELECT COUNT(*) FROM species_meta")
        if cur.fetchone()[0] == 0:
            species_meta_rows = [
                # COSTA
                ("Amancaes", "Amancaes", 8, 11, "sandy-loam; well-drained", "compost 3-5 kg/m2 annual; low N", "Prefers coastal dry soil; native; conserve water"),
                ("Huaranhuay", "Huaranhuay", 6, 9, "sandy-loam", "balanced compost; moderate P", "Resilient, tolerates saline influence"),
                ("Cardo costero", "Cardo costero", 7, 10, "sandy; well-drained", "organic mulch; low fertilizer", "Thorny coastal shrub"),
                ("Lirio de mar", "Lirio de mar", 9, 12, "sandy; slightly acidic", "seaweed compost; moderate K", "Coastal lily, needs good drainage"),

                # SIERRA
                ("Cantuta", "Cantuta", 11, 2, "loam; good drainage", "compost; moderate N", "Highland ornamental; cultural value"),
                ("Puya Raimondii", "Puya Raimondii", 12, 3, "rocky, poor soil", "minimal organic matter; avoid waterlogging", "Slow-growing giant bromeliad"),
                ("Kantu rosada", "Kantu rosada", 10, 1, "well-drained loam", "compost annually", "High-elevation ornamental"),
                ("Retama", "Retama", 6, 9, "sandy-loam; alkaline tolerant", "low organic requirement", "Resinous shrub adapted to poor soils"),
                ("Ichu", "Ichu", 1, 12, "coarse; poor; well-drained", "not required; light compost", "Native grass used for fodder and thatching"),
                ("MaracuyÃ¡", "MaracuyÃ¡", 9, 2, "rich loam; moist", "organic fertilizer; NPK balanced", "Passionfruit vine; requires staking and irrigation"),
                ("Flor de ichu", "Flor de ichu", 1, 12, "coarse; poor", "minimal", "High altitude grass flower"),

                # SELVA
                ("OrquÃ­deas", "OrquÃ­deas", 3, 8, "rich, humus; high organic", "orchid bark, moss; slow-release", "Many species; shade and humidity required"),
                ("Heliconia", "Heliconia", 5, 10, "rich loam; moist", "high-organic compost; mulching", "Tropical understory; needs humidity"),
                ("Achiote", "Achiote", 7, 11, "well-drained loam", "organic compost; phosphorus-rich", "Annatto tree; agroforestry-friendly"),
                ("Guaba", "Guaba", 9, 12, "loamy; nitrogen-fixing intercropping", "light compost", "Inga edulis, nitrogen-fixing"),
                ("OrquÃ­deas amazÃ³nicas", "OrquÃ­deas amazÃ³nicas", 3, 9, "epiphytic media or humus-rich soil", "orchid mixes; high humidity", "Diverse Amazon orchids"),
            ]
            cur.executemany("""
                INSERT INTO species_meta (name, common_name, bloom_start_month, bloom_end_month, soil, organic_recommendation, notes)
                VALUES (?,?,?,?,?,?,?)
            """, species_meta_rows)

        # seed regional_species if empty
        cur.execute("SELECT COUNT(*) FROM regional_species")
        if cur.fetchone()[0] == 0:
            regional_rows = [
                # COSTA
                ("Lima", None, "Amancaes"), ("Lima", None, "Huaranhuay"), ("Lima", None, "Cardo costero"),
                ("Ica", None, "Lirio de mar"), ("Ica", None, "Huaranhuay"),
                ("La Libertad", None, "Lirio de mar"), ("La Libertad", None, "Cardo costero"),
                ("Arequipa", None, "Retama"), ("Arequipa", None, "Huaranhuay"), ("Arequipa", None, "Cantuta"),
                # SIERRA
                ("Cusco", None, "Cantuta"), ("Cusco", None, "Puya Raimondii"), ("Cusco", None, "Kantu rosada"),
                ("Puno", None, "Puya Raimondii"), ("Puno", None, "Flor de ichu"),
                ("Ancash", None, "Cantuta"), ("Ancash", None, "Puya Raimondii"),
                ("Huancavelica", None, "Retama"), ("Huancavelica", None, "Cantuta"), ("Huancavelica", None, "Ichu"),
                ("JunÃ­n", None, "Retama"), ("JunÃ­n", None, "Ichu"), ("JunÃ­n", None, "MaracuyÃ¡"),
                ("Ayacucho", None, "Cantuta"), ("Ayacucho", None, "Puya Raimondii"), ("Ayacucho", None, "Retama"),
                # SELVA
                ("San MartÃ­n", None, "OrquÃ­deas"), ("San MartÃ­n", None, "Heliconia"), ("San MartÃ­n", None, "Achiote"),
                ("Loreto", None, "Heliconia"), ("Loreto", None, "Achiote"), ("Loreto", None, "Guaba"),
                ("Ucayali", None, "Heliconia"), ("Ucayali", None, "Achiote"), ("Ucayali", None, "MaracuyÃ¡"),
                ("Madre de Dios", None, "OrquÃ­deas"), ("Madre de Dios", None, "Heliconia"), ("Madre de Dios", None, "Achiote"),
                ("Amazonas", None, "OrquÃ­deas amazÃ³nicas"),
            ]
            cur.executemany("INSERT INTO regional_species (region, province, species_name) VALUES (?,?,?)", regional_rows)

        conn.commit()


# seed DB on import
init_db()


# --- Static root
@app.get("/")
async def root():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return JSONResponse({"error": "static/index.html not found", "static_dir": str(STATIC_DIR)}, status_code=500)
    return FileResponse(str(index_path))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/layers")
async def layers():
    return [
        {"id": "ndvi", "title": "Simulated NDVI", "type": "raster"},
        {"id": "stations", "title": "Seasons", "type": "vector"},
        {"id": "areas", "title": "Protected Areas", "type": "vector"},
        {"id": "crops", "title": "Crops", "type": "vector"},
        {"id": "pollinators", "title": "Pollinators", "type": "vector"},
        {"id": "species", "title": "Regional species", "type": "vector"},
    ]


# --- helpers
def rows_to_geojson(rows, geom_key=None, lon_key="lon", lat_key="lat", props=None):
    features = []
    for r in rows:
        props_dict = {}
        if props:
            for p in props:
                props_dict[p] = r[p]
        if geom_key and r[geom_key]:
            features.append({"type": "Feature", "geometry": None, "properties": {"wkt": r[geom_key], **props_dict}})
        elif lon_key in r.keys() and lat_key in r.keys():
            features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [r[lon_key], r[lat_key]]}, "properties": props_dict})
        else:
            features.append({"type": "Feature", "geometry": None, "properties": props_dict})
    return {"type": "FeatureCollection", "features": features}


# db endpoints
@app.get("/db/stations")
async def db_stations():
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id,name,lon,lat FROM stations")
        rows = cur.fetchall()
    return rows_to_geojson(rows, lon_key="lon", lat_key="lat", props=["id", "name"])


@app.get("/db/areas")
async def db_areas():
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id,name,geom FROM areas")
        rows = cur.fetchall()
    return rows_to_geojson(rows, geom_key="geom", props=["id", "name"])


@app.get("/db/crops")
async def db_crops():
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id,name,lon,lat,area FROM crops")
        rows = cur.fetchall()
    return rows_to_geojson(rows, lon_key="lon", lat_key="lat", props=["id", "name", "area"])


@app.get("/db/pollinators")
async def db_pollinators():
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id,name,lon,lat,abundance FROM pollinators")
        rows = cur.fetchall()
    return rows_to_geojson(rows, lon_key="lon", lat_key="lat", props=["id", "name", "abundance"])


# --- Geo / tile helpers
def tile_xyz_to_bbox(z: int, x: int, y: int) -> Tuple[float, float, float, float]:
    n = 2.0 ** z
    lon_left = x / n * 360.0 - 180.0
    lon_right = (x + 1) / n * 360.0 - 180.0
    lat_top = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_bottom = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return lon_left, lat_bottom, lon_right, lat_top


def perceptual_colormap(ndvi: np.ndarray) -> np.ndarray:
    v = np.clip((ndvi - NDVI_MIN_DISPLAY) / (NDVI_MAX_DISPLAY - NDVI_MIN_DISPLAY), 0.0, 1.0)
    stops = [
        (0.0, np.array([120, 63, 32], dtype=np.float32)),
        (0.25, np.array([255, 213, 79], dtype=np.float32)),
        (0.5, np.array([173, 239, 127], dtype=np.float32)),
        (0.75, np.array([60, 160, 80], dtype=np.float32)),
        (1.0, np.array([10, 90, 40], dtype=np.float32)),
    ]
    rgb = np.zeros(ndvi.shape + (3,), dtype=np.uint8)
    for i in range(len(stops) - 1):
        lpos, lcol = stops[i]; rpos, rcol = stops[i + 1]
        mask = (v >= lpos) & (v <= rpos)
        if np.any(mask):
            t = (v[mask] - lpos) / (rpos - lpos)
            interp = (1 - t)[:, None] * lcol[None, :] + t[:, None] * rcol[None, :]
            rgb[mask] = np.round(interp).astype(np.uint8)
    rgb[v <= stops[0][0]] = stops[0][1]; rgb[v >= stops[-1][0]] = stops[-1][1]
    return rgb


def deterministic_rng_seed(key: str) -> np.random.RandomState:
    h = hashlib.sha256(key.encode("utf-8")).digest()[:4]
    seed = int.from_bytes(h, "big")
    return np.random.RandomState(seed)


def generate_ndvi_array(tile_bbox: Tuple[float, float, float, float], z: int, x: int, y: int, date_str: str, params=None) -> np.ndarray:
    if params is None:
        params = SIM_PARAMS
    lon0, lat0, lon1, lat1 = tile_bbox
    xs = np.linspace(lon0, lon1, TILE_SIZE)
    ys = np.linspace(lat1, lat0, TILE_SIZE)
    lon_grid, lat_grid = np.meshgrid(xs, ys)
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        dt = datetime.utcnow()
    doy = dt.timetuple().tm_yday
    seasonal = params["seasonal_amp"] * (0.5 * (1 + np.sin(2 * math.pi * (doy / 365.0))))
    lat_factor = -0.004 * (lat_grid + 12.0)
    cx = (lon0 + lon1) / 2.0 + ((x % 4) - 1.5) * 0.018
    cy = (lat0 + lat1) / 2.0 + ((y % 4) - 1.5) * 0.018
    r = np.sqrt((lon_grid - cx) ** 2 + (lat_grid - cy) ** 2)
    radial = np.exp(- (r * 18.0) ** 2)
    seed_key = f"{z}-{x}-{y}-{date_str}"
    rng = deterministic_rng_seed(seed_key)
    noise = rng.normal(scale=params["noise_level"], size=(TILE_SIZE, TILE_SIZE))
    ndvi = seasonal + params["radial_amp"] * radial + lat_factor + noise
    ndvi = np.clip(ndvi, NDVI_MIN_DISPLAY, NDVI_MAX_DISPLAY)
    return ndvi


def cache_tile_path(date: str, z: int, x: int, y: int) -> str:
    ddir = os.path.join(TILE_CACHE_DIR, date)
    os.makedirs(ddir, exist_ok=True)
    return os.path.join(ddir, f"{z}_{x}_{y}.png")


def write_atomic(path: str, data: bytes):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


@app.get("/tiles/{z}/{x}/{y}.png")
async def get_tile(z: int, x: int, y: int, date: str = Query(None)):
    if date is None:
        date = datetime.utcnow().strftime("%Y-%m-%d")
    if z < 0 or x < 0 or y < 0:
        raise HTTPException(status_code=400, detail="Invalid tile coords")
    max_xy = 2 ** z
    if x >= max_xy or y >= max_xy:
        raise HTTPException(status_code=400, detail="Tile coords out of range for zoom")
    cached = cache_tile_path(date, z, x, y)
    if os.path.exists(cached):
        try:
            with open(cached, "rb") as fh:
                data = fh.read()
            return Response(content=data, media_type="image/png", headers={"X-Cache": "HIT"})
        except Exception:
            logger.exception("Failed reading cached tile; regenerating")
    bbox = tile_xyz_to_bbox(z, x, y)
    ndvi = generate_ndvi_array(bbox, z, x, y, date)
    rgb = perceptual_colormap(ndvi)
    veg_mask = ndvi >= VEG_ALPHA_THRESHOLD
    alpha = np.where(veg_mask, np.clip(((ndvi - NDVI_MIN_DISPLAY) / (NDVI_MAX_DISPLAY - NDVI_MIN_DISPLAY)) * 255, 70, 255), 0).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    data = buf.getvalue()
    try:
        write_atomic(cached, data)
    except Exception:
        logger.exception("Failed writing cached tile")
    return Response(content=data, media_type="image/png", headers={"X-Cache": "MISS"})


# --- Timeseries (deterministic RNG)
def point_timeseries_logic(lon: float, lat: float, points: int = 24, aggregate: str = "monthly", date: str = None):
    if date:
        try:
            end = datetime.strptime(date, "%Y-%m-%d")
        except Exception:
            end = datetime.utcnow()
    else:
        end = datetime.utcnow()
    if aggregate == "monthly":
        months = []
        dt = end.replace(day=1)
        for i in range(points):
            months.append(dt.strftime("%Y-%m"))
            dt = (dt.replace(day=1) - timedelta(days=1)).replace(day=1)
        months = list(reversed(months))
        vals = []
        seed_key = f"pt-{lon:.6f}-{lat:.6f}-{date or ''}"
        rng = deterministic_rng_seed(seed_key)
        for i, m in enumerate(months):
            base = 0.25 + 0.12 * math.sin(2 * math.pi * (i / 12.0))
            noise = float(rng.normal(scale=0.04))
            vals.append(round(float(np.clip(base + noise, NDVI_MIN_DISPLAY, NDVI_MAX_DISPLAY)), 4))
        return {"dates": months, "values": vals}
    now = end
    dates = [(now - timedelta(days=i)).strftime("%Y-%m-%d") for i in reversed(range(points))]
    seed_key = f"ts-{lon:.6f}-{lat:.6f}-{date or ''}"
    rng = deterministic_rng_seed(seed_key)
    values = []
    for i in range(points):
        d = now - timedelta(days=points - 1 - i)
        doy = d.timetuple().tm_yday
        base = 0.25 * (0.5 * (1 + math.sin(2 * math.pi * (doy / 365.0))))
        noise = rng.normal(scale=0.03)
        val = np.clip(base + noise, NDVI_MIN_DISPLAY, NDVI_MAX_DISPLAY)
        values.append(round(float(val), 4))
    return {"dates": dates, "values": values}


@app.get("/point-timeseries")
async def point_timeseries(lon: float, lat: float, points: int = 24, aggregate: str = "monthly", date: str = None):
    points = max(1, min(points, 720))
    return point_timeseries_logic(lon, lat, points=points, aggregate=aggregate, date=date)


# --- AOI helpers (sampling)
def polygon_bounds_from_wkt(wkt: str):
    txt = wkt.replace("POLYGON((", "").replace("))", "").strip()
    coords = [tuple(map(float, p.split())) for p in txt.split(",")]
    lons = [c[0] for c in coords]; lats = [c[1] for c in coords]
    return min(lons), min(lats), max(lons), max(lats)


def sample_ndvi_over_bbox(lon_min, lat_min, lon_max, lat_max, date: str, sample=128):
    xs = np.linspace(lon_min, lon_max, sample); ys = np.linspace(lat_max, lat_min, sample)
    lon_grid, lat_grid = np.meshgrid(xs, ys)
    vals = np.full(lon_grid.shape, np.nan, dtype=np.float32)
    # simple reuse strategy: cache by z/x/y when sampling (z fixed at 10)
    z = 10
    n = 2 ** z
    tile_cache: Dict[Tuple[int, int, int], np.ndarray] = {}
    for i in range(sample):
        for j in range(sample):
            lon = lon_grid[i, j]; lat = lat_grid[i, j]
            x = int((lon + 180.0) / 360.0 * n)
            lat_rad = math.radians(lat)
            y = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
            key = (z, x, y)
            if key not in tile_cache:
                bbox = tile_xyz_to_bbox(z, x, y)
                tile_cache[key] = generate_ndvi_array(bbox, z, x, y, date)
            ndvi_tile = tile_cache[key]
            bbox = tile_xyz_to_bbox(z, x, y)
            px = int((lon - bbox[0]) / (bbox[2] - bbox[0]) * TILE_SIZE)
            py = int((bbox[3] - lat) / (bbox[3] - bbox[1]) * TILE_SIZE)
            px = np.clip(px, 0, TILE_SIZE - 1); py = np.clip(py, 0, TILE_SIZE - 1)
            vals[i, j] = ndvi_tile[py, px]
    return vals.flatten()


@app.get("/areas/{area_id}/ndvi-summary")
async def area_ndvi_summary(area_id: int, date: str = Query(None)):
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id,name,geom FROM areas WHERE id=?", (area_id,))
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Area not found")
    if date is None:
        date = datetime.utcnow().strftime("%Y-%m-%d")
    lon_min, lat_min, lon_max, lat_max = polygon_bounds_from_wkt(row["geom"])
    samples = sample_ndvi_over_bbox(lon_min, lat_min, lon_max, lat_max, date, sample=64)
    mean = float(np.nanmean(samples)); median = float(np.nanmedian(samples)); std = float(np.nanstd(samples))
    p10 = float(np.percentile(samples, 10)); p90 = float(np.percentile(samples, 90))
    return {"area_id": area_id, "date": date, "mean": mean, "median": median, "std": std, "p10": p10, "p90": p90}


def area_ndvi_timeseries_logic(geom_wkt: str, start_dt: datetime, end_dt: datetime):
    dates = [(start_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end_dt - start_dt).days + 1)]
    lon_min, lat_min, lon_max, lat_max = polygon_bounds_from_wkt(geom_wkt)
    values = []
    for d in dates:
        samples = sample_ndvi_over_bbox(lon_min, lat_min, lon_max, lat_max, d, sample=40)
        values.append(round(float(np.nanmean(samples)), 4))
    return {"dates": dates, "values": values}


@app.get("/areas/{area_id}/ndvi-timeseries")
async def area_ndvi_timeseries(area_id: int, start: str = Query(None), end: str = Query(None), points: int = 30):
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id,name,geom FROM areas WHERE id=?", (area_id,))
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Area not found")
    if end is None:
        end_dt = datetime.utcnow()
    else:
        end_dt = datetime.strptime(end, "%Y-%m-%d")
    if start is None:
        start_dt = end_dt - timedelta(days=points - 1)
    else:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
    return area_ndvi_timeseries_logic(row["geom"], start_dt, end_dt)


@app.get("/compare")
async def compare_areas(area1: int = Query(...), area2: int = Query(...), start: str = Query(None), end: str = Query(None)):
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id,name,geom FROM areas WHERE id=?", (area1,))
        r1 = cur.fetchone()
        cur.execute("SELECT id,name,geom FROM areas WHERE id=?", (area2,))
        r2 = cur.fetchone()
    if not r1 or not r2:
        raise HTTPException(status_code=404, detail="Area not found")
    if end is None:
        end_dt = datetime.utcnow()
    else:
        end_dt = datetime.strptime(end, "%Y-%m-%d")
    if start is None:
        start_dt = end_dt - timedelta(days=29)
    else:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
    a1 = area_ndvi_timeseries_logic(r1["geom"], start_dt, end_dt)
    a2 = area_ndvi_timeseries_logic(r2["geom"], start_dt, end_dt)
    if len(a1["dates"]) != len(a2["dates"]):
        start_dt = datetime.strptime(a1["dates"][0], "%Y-%m-%d")
        end_dt = datetime.strptime(a1["dates"][-1], "%Y-%m-%d")
        dates = [(start_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end_dt - start_dt).days + 1)]
    else:
        dates = a1["dates"]
    diffs = [round(a1["values"][i] - a2["values"][i], 4) for i in range(len(dates))]
    return {"area1": area1, "area2": area2, "dates": dates, "values1": a1["values"], "values2": a2["values"], "diffs": diffs}


# --- Phenology helpers
def detect_peak_and_onset(values: List[float], dates: List[str], threshold: float = 0.5):
    if not values or not dates:
        return {"peak_date": None, "peak_value": None, "onset_date": None}
    peak_idx = int(np.argmax(values))
    peak_date = dates[peak_idx] if len(dates) > 0 else None
    onset_idx = next((i for i, v in enumerate(values) if v >= threshold), None)
    onset_date = dates[onset_idx] if onset_idx is not None else None
    return {"peak_date": peak_date, "peak_value": values[peak_idx] if len(values) > 0 else None, "onset_date": onset_date}


@app.get("/areas/{area_id}/phenology")
async def area_phenology(area_id: int, start: str = Query(None), end: str = Query(None)):
    ts = await area_ndvi_timeseries(area_id, start=start, end=end)
    info = detect_peak_and_onset(ts["values"], ts["dates"], threshold=0.5)
    return {"area_id": area_id, "phenology": info, "timeseries": ts}


@app.get("/areas/rank")
async def areas_rank_by_growth(days: int = Query(14)):
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id,name,geom FROM areas")
        rows = cur.fetchall()
    results = []
    for r in rows:
        area_id = r["id"]
        end = datetime.utcnow(); start = end - timedelta(days=days)
        ts = area_ndvi_timeseries_logic(r["geom"], start, end)
        if len(ts["values"]) < 2:
            continue
        growth = ts["values"][-1] - ts["values"][0]
        results.append({"area_id": area_id, "name": r["name"], "growth": round(growth, 4)})
    results.sort(key=lambda x: x["growth"], reverse=True)
    return results


# --- New species endpoints
@app.get("/db/species-by-region/{region}")
async def species_by_region(region: str):
    region_q = region.strip()
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT species_name FROM regional_species WHERE LOWER(region)=LOWER(?)", (region_q,))
        rows = cur.fetchall()
    species = [r[0] for r in rows]
    return {"region": region_q, "species": species}


@app.get("/species/{name}")
async def species_meta(name: str):
    nm = name.strip()
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT name, common_name, bloom_start_month, bloom_end_month, soil, organic_recommendation, notes FROM species_meta WHERE LOWER(name)=LOWER(?)", (nm,))
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Species not found")
    data = {
        "name": row["name"],
        "common_name": row["common_name"],
        "bloom_start_month": int(row["bloom_start_month"]) if row["bloom_start_month"] is not None else None,
        "bloom_end_month": int(row["bloom_end_month"]) if row["bloom_end_month"] is not None else None,
        "soil": row["soil"],
        "organic_recommendation": row["organic_recommendation"],
        "notes": row["notes"],
    }
    return data


# --- Region page + API
@app.get("/regions/{region}", response_class=HTMLResponse)
async def region_page(region: str):
    page = STATIC_DIR / "region.html"
    if not page.exists():
        return HTMLResponse("<h1>Region page not found</h1>", status_code=404)
    return FileResponse(str(page))


@app.get("/api/region/{region}")
async def api_region_info(region: str):
    region_q = region.strip()
    with get_db_conn() as conn:
        cur = conn.cursor()
        # species names for region
        cur.execute("SELECT DISTINCT species_name FROM regional_species WHERE LOWER(region)=LOWER(?)", (region_q,))
        species_rows = cur.fetchall()
        species = [r[0] for r in species_rows]

        # fetch metadata for each species (if available)
        species_meta_list = []
        if species:
            placeholders = ",".join("?" for _ in species)
            # build case-insensitive match list
            lowered = [s.lower() for s in species]
            # fetch all species_meta and match in-Python to avoid SQL dialect issues
            cur.execute("SELECT name, common_name, bloom_start_month, bloom_end_month, soil, organic_recommendation, notes FROM species_meta")
            meta_rows = cur.fetchall()
            meta_map = {row["name"].lower(): dict(row) for row in meta_rows}
            for s in species:
                m = meta_map.get(s.lower())
                if m:
                    species_meta_list.append({
                        "name": m["name"],
                        "common_name": m["common_name"],
                        "bloom_start_month": int(m["bloom_start_month"]) if m["bloom_start_month"] is not None else None,
                        "bloom_end_month": int(m["bloom_end_month"]) if m["bloom_end_month"] is not None else None,
                        "soil": m["soil"],
                        "organic_recommendation": m["organic_recommendation"],
                        "notes": m["notes"],
                    })
                else:
                    species_meta_list.append({"name": s})
        # simple counts
        cur.execute("SELECT COUNT(*) FROM regional_species WHERE LOWER(region)=LOWER(?)", (region_q,))
        species_count = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM areas WHERE LOWER(name) LIKE LOWER(?)", (f"%{region_q}%",))
        area_ref_count = int(cur.fetchone()[0])

    return {"region": region_q, "species_count": species_count, "area_ref_count": area_ref_count, "species": species_meta_list}


# --- Entrypoint note
# You run uvicorn with:
# uvicorn main:app --reload --host 127.0.0.1 --port 8080
# This file is robust to being started from any CWD because os.chdir(BASE_DIR) is called near the top.

if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True)
