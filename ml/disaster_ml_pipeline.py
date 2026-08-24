"""
Raksha Grid — Multi-Hazard Disaster Risk Zoning ML Pipeline
=============================================================

Goal
----
Given a STATE and a YEAR, produce a grid of geo-tagged risk zones
(safe / low / moderate / severe) for every hazard relevant to that
region, so the frontend can shade a map like a weather radar.

Design decision: one model per hazard, not one model for everything
-----------------------------------------------------------------
Floods, landslides, avalanches, sandstorms and cyclones are driven
by almost entirely different physical processes and different
feature sets. A single "disaster classifier" would force irrelevant
features onto every hazard (e.g. snowpack for a Rajasthan sandstorm)
and blur feature importance. So each hazard gets its own model,
trained only on the features that are physically relevant to it,
and a shared pipeline (features -> train -> zone) is reused across
all of them.

Data reality check (read before trusting "100 years of data")
---------------------------------------------------------------
True wall-to-wall, grid-resolution, 100-year records do not exist
for most of these variables in India. What actually exists:

  - Rainfall            : IMD gridded data, reliable from ~1901 (0.25 deg)
  - River discharge      : CWC gauge data, dense only from ~1970s
  - Soil moisture         : satellite-era only, reliable from ~1980s (ESA CCI)
  - Vegetation / deforestation : satellite-era only, from ~1972 (Landsat), robust from ~2000 (MODIS)
  - Snowpack / avalanche : sparse before ~1990s; IMD/DGRE stations limited
  - Cyclone tracks        : IMD best-track data, reliable from ~1890 for
                             landfall/intensity, less reliable for finer detail pre-1970
  - Historical disaster event labels (for supervised learning) : EM-DAT (1900+,
    but under-reports before ~1960), NDMA/state DMA records (post-2005 are
    most granular)

Practical implication for the pipeline: earlier decades (pre-1970) get
fewer, coarser features and are weighted less in training (see
`era_weight` in `engineer_features`). This script is written against
realistic column names so it's a straight swap once real feeds are
plugged in; the `generate_synthetic_history` function is ONLY a stand-in
so the rest of the pipeline can be built, run, and demoed end-to-end
today.
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import state_geo

RNG = np.random.default_rng(42)

# =============================================================================
# 1. CONFIG — which hazards apply where, and what features drive each hazard
# =============================================================================

# Real source each feature would come from in production (kept as comments
# next to each field so swapping synthetic -> real data is a find/replace job).
HAZARD_FEATURES: Dict[str, List[str]] = {
    "flood": [
        "rainfall_7day_mm",        # IMD gridded rainfall, 7-day rolling sum
        "rainfall_anomaly_pct",    # vs 30-yr seasonal normal, IMD
        "river_discharge_cumecs",  # CWC gauge stations
        "soil_saturation_index",   # ESA CCI soil moisture
        "elevation_m",             # SRTM DEM (static)
        "slope_deg",               # derived from DEM (static)
        "drainage_density",        # river network density, static, from CWC/Bhuvan
        "urbanization_pct",        # impervious surface %, satellite land-use
        "upstream_dam_release",    # CWC / state irrigation dept
    ],
    "landslide": [
        "rainfall_24hr_mm",        # IMD — short, intense rainfall is the trigger
        "rainfall_antecedent_mm",  # prior 15-day rainfall (saturates the slope)
        "slope_deg",               # DEM derived (static)
        "soil_type_erodibility",   # NBSS&LUP soil maps (static)
        "forest_cover_pct",        # tree cover — roots stabilize slope, Forest Survey of India
        "deforestation_trend",     # % forest lost over trailing 10 yrs, satellite
        "seismic_activity_index",  # NCS recent tremor energy (shaking loosens slope)
        "road_cut_density",        # construction/road cuts destabilize slope
    ],
    "avalanche": [
        "snowpack_depth_cm",       # DGRE / IMD Himalayan stations
        "new_snow_24hr_cm",        # DGRE
        "temperature_swing_c",     # freeze-thaw cycles, IMD
        "slope_deg",               # DEM (static) — 30-45 deg = highest risk band
        "aspect",                  # slope-facing direction (N-facing holds snow longer)
        "wind_loading_index",      # wind-transported snow accumulation
    ],
    "sandstorm": [
        "wind_speed_kmh",          # IMD
        "aridity_index",           # precip / potential evapotranspiration, long-term
        "soil_moisture_pct",       # ESA CCI
        "vegetation_cover_pct",    # NDVI, satellite
        "temperature_c",           # IMD
        "land_degradation_index",  # desertification trend, ISRO/Bhuvan
    ],
    "cyclone": [
        "sea_surface_temp_c",      # NOAA / INCOIS — cyclones need warm water (>26.5C)
        "distance_to_coast_km",    # static
        "historical_landfall_freq",# IMD best-track climatology for this coastal segment
        "central_pressure_hpa",    # IMD track data, storm-specific
        "wind_speed_kmh",          # IMD track data, storm-specific
        "elevation_m",             # static — low-lying = higher surge risk
    ],
}

STATE_HAZARDS: Dict[str, List[str]] = {
    # Himalayan belt — landslide + avalanche (high altitude) + flood (valleys)
    "Jammu and Kashmir": ["avalanche", "landslide", "flood"],
    "Jammu & Kashmir": ["avalanche", "landslide", "flood"],
    "Ladakh": ["avalanche"],
    "Himachal Pradesh": ["landslide", "avalanche", "flood"],
    "Uttarakhand": ["landslide", "avalanche", "flood"],
    "Sikkim": ["landslide", "avalanche"],
    # North-east — flood + landslide (hilly terrain, monsoon rainfall)
    "Assam": ["flood", "landslide"],
    "Arunachal Pradesh": ["landslide", "flood"],
    "Meghalaya": ["flood", "landslide"],
    "Manipur": ["landslide", "flood"],
    "Mizoram": ["landslide", "flood"],
    "Nagaland": ["landslide", "flood"],
    "Tripura": ["flood", "landslide"],
    # Indo-Gangetic plain — river flood dominant
    "Bihar": ["flood"],
    "Uttar Pradesh": ["flood"],
    "Jharkhand": ["flood"],
    "West Bengal": ["cyclone", "flood"],
    "Madhya Pradesh": ["flood"],
    "Chhattisgarh": ["flood"],
    "NCT of Delhi": ["flood"],
    "Delhi": ["flood"],
    "Chandigarh": ["flood"],
    "Punjab": ["flood", "sandstorm"],
    "Haryana": ["flood", "sandstorm"],
    # Arid west — sandstorm dominant
    "Rajasthan": ["sandstorm"],
    "Gujarat": ["sandstorm", "cyclone"],
    "Dadara & Nagar Havelli": ["flood"],
    "Dadra and Nagar Haveli and Daman and Diu": ["cyclone", "flood"],
    "Daman & Diu": ["cyclone"],
    # Coastal peninsula — cyclone + flood
    "Odisha": ["cyclone", "flood"],
    "Andhra Pradesh": ["cyclone", "flood"],
    "Telangana": ["flood"],
    "Tamil Nadu": ["cyclone", "flood"],
    "Puducherry": ["cyclone", "flood"],
    "Kerala": ["flood", "landslide", "cyclone"],
    "Karnataka": ["flood", "cyclone"],
    "Goa": ["cyclone", "flood"],
    "Maharashtra": ["flood", "cyclone"],
    # Islands — cyclone dominant
    "Andaman & Nicobar Island": ["cyclone"],
    "Andaman and Nicobar Islands": ["cyclone"],
    "Lakshadweep": ["cyclone"],
}

SEVERITY_LABELS = ["safe", "low", "moderate", "severe"]
GRID_RESOLUTION_DEG = 0.25  # ~25km cells — matches IMD gridded rainfall resolution


# =============================================================================
# 2. SYNTHETIC HISTORICAL DATASET (stand-in for the real 100-yr feeds above)
# =============================================================================

def build_grid(state: str, resolution: float = None) -> pd.DataFrame:
    """Static grid of cells covering a state's REAL shape (not a rectangle).

    Uses state_geo.cells_in_state(), which loads the actual state polygon
    (from india_states.geojson) and keeps only grid-cell centers that fall
    inside it — so a state like Kerala or Assam gets a grid that follows
    its real coastline/border, not a bounding box that spills into
    neighboring states or the sea.
    """
    if resolution is None:
        resolution = state_geo.pick_resolution(state)
    raw_cells = state_geo.cells_in_state(state, resolution)
    cells = [
        {"cell_id": f"{state[:3].upper()}-{i}", "lat": lat, "lon": lon}
        for i, (lat, lon) in enumerate(raw_cells)
    ]
    grid = pd.DataFrame(cells)
    if grid.empty:
        raise ValueError(f"No grid cells generated for '{state}' — check state_geo boundary data.")

    is_himalayan = state in (
        "Jammu and Kashmir", "Jammu & Kashmir", "Ladakh", "Himachal Pradesh",
        "Uttarakhand", "Sikkim", "Arunachal Pradesh",
    )
    # static terrain features, seeded per-cell so they're stable across years
    grid["elevation_m"] = RNG.uniform(500, 6000, len(grid)) if is_himalayan \
        else RNG.uniform(5, 600, len(grid))
    grid["slope_deg"] = RNG.uniform(0, 45, len(grid))

    min_lon, min_lat, max_lon, max_lat = state_geo.get_bounds(state)
    # crude proxy for "how close to the coast" -- real pipeline would use
    # an actual coastline distance raster (e.g. from Bhuvan/NOAA)
    grid["distance_to_coast_km"] = RNG.uniform(0, 400, len(grid))
    return grid


def generate_synthetic_history(state: str, start_year=1925, end_year=2025) -> pd.DataFrame:
    """
    Simulates a plausible 100-year multi-hazard record for one state.
    STAND-IN ONLY — see module docstring. In production this function is
    replaced by ETL jobs pulling IMD/CWC/NCS/DGRE/Forest Survey/EM-DAT data
    and joining it onto `build_grid(state)` by cell + year.
    """
    grid = build_grid(state)
    years = np.arange(start_year, end_year + 1)
    hazards = STATE_HAZARDS[state]
    rows = []

    for year in years:
        # data richness improves over time — see docstring on data reality
        era_weight = np.clip((year - 1925) / (2025 - 1925), 0.15, 1.0)
        monsoon_strength = RNG.normal(1.0, 0.25)  # year-to-year monsoon variability

        for _, cell in grid.iterrows():
            record = {
                "state": state, "year": int(year), "cell_id": cell.cell_id,
                "lat": cell.lat, "lon": cell.lon, "era_weight": era_weight,
                "elevation_m": cell.elevation_m, "slope_deg": cell.slope_deg,
                "distance_to_coast_km": cell.distance_to_coast_km,
            }

            if "flood" in hazards:
                rainfall = max(0, RNG.normal(180, 60) * monsoon_strength)
                record.update({
                    "rainfall_7day_mm": rainfall,
                    "rainfall_anomaly_pct": (monsoon_strength - 1) * 100,
                    "river_discharge_cumecs": max(0, rainfall * 8 + RNG.normal(0, 200)),
                    "soil_saturation_index": np.clip(rainfall / 300 + RNG.normal(0, 0.1), 0, 1),
                    "drainage_density": RNG.uniform(0.1, 2.5),
                    "urbanization_pct": np.clip((year - 1925) / 150 + RNG.uniform(0, 0.2), 0, 1) * 100,
                    "upstream_dam_release": RNG.uniform(0, 500) if year > 1960 else 0,
                })
                flood_risk = (
                    0.4 * np.clip(rainfall / 350, 0, 1)
                    + 0.3 * record["soil_saturation_index"]
                    + 0.2 * np.clip(record["river_discharge_cumecs"] / 3000, 0, 1)
                    - 0.15 * np.clip(cell.elevation_m / 500, 0, 1)
                )
                record["flood_severity"] = _to_severity(flood_risk)

            if "landslide" in hazards:
                rain24 = max(0, RNG.normal(60, 30) * monsoon_strength)
                forest_cover = np.clip(70 - (year - 1925) * 0.15 + RNG.normal(0, 8), 10, 90)
                record.update({
                    "rainfall_24hr_mm": rain24,
                    "rainfall_antecedent_mm": max(0, RNG.normal(200, 80) * monsoon_strength),
                    "soil_type_erodibility": RNG.uniform(0.2, 0.9),
                    "forest_cover_pct": forest_cover,
                    "deforestation_trend": np.clip(RNG.normal(0.5, 1.5), -2, 5),
                    "seismic_activity_index": RNG.exponential(0.3),
                    "road_cut_density": np.clip((year - 1950) / 75, 0, 1) * RNG.uniform(0, 1),
                })
                slope_factor = np.clip((cell.slope_deg - 15) / 30, 0, 1)
                landslide_risk = (
                    0.35 * slope_factor
                    + 0.25 * np.clip(rain24 / 150, 0, 1)
                    + 0.2 * (1 - forest_cover / 90)
                    + 0.2 * np.clip(record["seismic_activity_index"], 0, 1)
                )
                record["landslide_severity"] = _to_severity(landslide_risk)

            if "avalanche" in hazards:
                snowpack = max(0, RNG.normal(150, 60))
                record.update({
                    "snowpack_depth_cm": snowpack,
                    "new_snow_24hr_cm": max(0, RNG.normal(15, 10)),
                    "temperature_swing_c": RNG.uniform(2, 20),
                    "aspect": RNG.choice(["N", "S", "E", "W"]),
                    "wind_loading_index": RNG.uniform(0, 1),
                })
                slope_band = 1 - abs(cell.slope_deg - 37) / 37  # 30-45deg = peak risk
                avalanche_risk = (
                    0.4 * np.clip(slope_band, 0, 1)
                    + 0.3 * np.clip(snowpack / 300, 0, 1)
                    + 0.3 * np.clip(record["temperature_swing_c"] / 20, 0, 1)
                ) if cell.elevation_m > 2000 else 0.0
                record["avalanche_severity"] = _to_severity(avalanche_risk)

            if "sandstorm" in hazards:
                aridity = np.clip(RNG.normal(0.6, 0.15), 0, 1)
                veg = np.clip(30 - (year - 1925) * 0.05 + RNG.normal(0, 10), 2, 60)
                record.update({
                    "wind_speed_kmh": max(0, RNG.normal(35, 15)),
                    "aridity_index": aridity,
                    "soil_moisture_pct": np.clip(1 - aridity + RNG.normal(0, 0.1), 0, 1) * 100,
                    "vegetation_cover_pct": veg,
                    "temperature_c": RNG.normal(38, 6),
                    "land_degradation_index": np.clip(aridity - veg / 100, 0, 1),
                })
                sandstorm_risk = (
                    0.35 * aridity
                    + 0.35 * np.clip(record["wind_speed_kmh"] / 70, 0, 1)
                    + 0.3 * (1 - veg / 60)
                )
                record["sandstorm_severity"] = _to_severity(sandstorm_risk)

            if "cyclone" in hazards:
                sst = RNG.normal(28, 1.5)
                landfall_freq = np.clip(1 - cell.distance_to_coast_km / 200, 0, 1)
                record.update({
                    "sea_surface_temp_c": sst,
                    "historical_landfall_freq": landfall_freq,
                    "central_pressure_hpa": RNG.normal(985, 15),
                    "wind_speed_kmh": max(0, RNG.normal(90, 40) * landfall_freq),
                })
                cyclone_risk = (
                    0.4 * np.clip((sst - 26) / 4, 0, 1)
                    + 0.4 * landfall_freq
                    + 0.2 * np.clip(record["wind_speed_kmh"] / 180, 0, 1)
                )
                record["cyclone_severity"] = _to_severity(cyclone_risk)

            rows.append(record)

    return pd.DataFrame(rows)


def _to_severity(risk_score: float) -> str:
    risk_score = float(np.clip(risk_score + RNG.normal(0, 0.05), 0, 1))  # observation noise
    if risk_score < 0.25:
        return "safe"
    if risk_score < 0.5:
        return "low"
    if risk_score < 0.75:
        return "moderate"
    return "severe"


# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================

def engineer_features(df: pd.DataFrame, hazard: str) -> pd.DataFrame:
    """
    Adds derived features on top of the raw columns:
      - lag features (last year's value at the same cell -> persistence signal)
      - rolling multi-year trend (5-yr mean -> smooths noisy single-year spikes)
      - state-normalized z-scores (so a model trained across states isn't
        dominated by absolute-scale differences, e.g. Assam vs Rajasthan rainfall)
    """
    df = df.sort_values(["cell_id", "year"]).copy()
    feature_cols = HAZARD_FEATURES[hazard]

    for col in feature_cols:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        df[f"{col}_lag1"] = df.groupby("cell_id")[col].shift(1)
        df[f"{col}_5yr_mean"] = (
            df.groupby("cell_id")[col].transform(lambda s: s.rolling(5, min_periods=1).mean())
        )
        df[f"{col}_zscore"] = (
            df.groupby("state")[col].transform(lambda s: (s - s.mean()) / (s.std() + 1e-6))
        )

    df = df.fillna(0)
    return df


# =============================================================================
# 4. MODEL — one HistGradientBoostingClassifier per hazard
# =============================================================================
#
# Why gradient-boosted trees and not a neural net:
#   - Tabular, heterogeneous features (mm, degrees, %, indices) with no
#     spatial/sequential structure a CNN/RNN could exploit at this scale.
#   - Small-to-medium N (thousands to low millions of cell-years), where
#     boosted trees consistently beat deep nets in published tabular
#     benchmarks.
#   - Naturally handles missing values (`HistGradientBoostingClassifier`
#     splits on missingness directly) — important since older-era records
#     are sparse (see data reality note).
#   - Gives feature importances / SHAP values for free, which matters a lot
#     when the output is safety-critical: a coordinator should be able to
#     ask "why did this cell get flagged severe?" and get a real answer.
#
# Multi-class target: predicting the 4-level severity bin directly
# (safe/low/moderate/severe) rather than a raw regression score, because
# the frontend and NDRF consumers act on the bin, not the exact float —
# and classification gives calibrated class probabilities we can use as a
# confidence overlay.

@dataclass
class HazardModel:
    hazard: str
    model: HistGradientBoostingClassifier = field(default=None)
    label_encoder: LabelEncoder = field(default_factory=LabelEncoder)
    feature_cols: List[str] = field(default_factory=list)


def train_hazard_model(df: pd.DataFrame, hazard: str, test_year_cutoff: int = 2015, verbose: bool = True) -> HazardModel:
    """
    Time-based split (not random shuffling): train on years before the
    cutoff, test on years after. Random shuffling would leak future
    information into training via the lag/rolling features and give a
    falsely optimistic score — a model that "predicts the past" is
    useless operationally.
    """
    target_col = f"{hazard}_severity"
    df = df[df[target_col].notna()].copy()

    feature_cols = [c for c in df.columns if c not in (
        "state", "year", "cell_id", "lat", "lon", target_col
    ) and pd.api.types.is_numeric_dtype(df[c])]

    train_df = df[df.year < test_year_cutoff]
    test_df = df[df.year >= test_year_cutoff]

    le = LabelEncoder().fit(SEVERITY_LABELS)
    y_train = le.transform(train_df[target_col])
    y_test = le.transform(test_df[target_col])

    # older-era rows get down-weighted (see era_weight / data reality note)
    sample_weight = train_df["era_weight"].values if "era_weight" in train_df else None

    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=6,
        l2_regularization=0.5, random_state=42,
    )
    model.fit(train_df[feature_cols], y_train, sample_weight=sample_weight)

    preds = model.predict(test_df[feature_cols])
    if verbose:
        print(f"\n[{hazard}] hold-out report (years >= {test_year_cutoff}):")
        print(classification_report(
            y_test, preds, labels=range(len(le.classes_)),
            target_names=le.classes_, zero_division=0,
        ))
    try:
        proba = model.predict_proba(test_df[feature_cols])
        auc = roc_auc_score(y_test, proba, multi_class="ovr", labels=range(len(le.classes_)))
        if verbose:
            print(f"[{hazard}] macro OVR AUC: {auc:.3f}")
    except ValueError as e:
        if verbose:
            print(f"[{hazard}] AUC skipped (hold-out set too small/imbalanced): {e}")

    return HazardModel(hazard=hazard, model=model, label_encoder=le, feature_cols=feature_cols)


# =============================================================================
# 5. ZONING — turn per-cell predictions into a GeoJSON the frontend can shade
# =============================================================================

def predict_zones(hazard_model: HazardModel, df_year: pd.DataFrame, state: str = None, resolution: float = None) -> dict:
    """Runs the model for one (state, year) slice and returns a GeoJSON
    FeatureCollection of square cells colored by predicted severity, with
    the class probability included so the frontend can show confidence."""
    if resolution is None:
        resolution = state_geo.pick_resolution(state) if state else GRID_RESOLUTION_DEG
    X = df_year[hazard_model.feature_cols]
    pred_idx = hazard_model.model.predict(X)
    proba = hazard_model.model.predict_proba(X)
    severities = hazard_model.label_encoder.inverse_transform(pred_idx)
    confidences = proba.max(axis=1)

    features = []
    half = resolution / 2
    for (_, row), severity, conf in zip(df_year.iterrows(), severities, confidences):
        lat, lon = row.lat, row.lon
        polygon = [[
            [lon - half, lat - half], [lon + half, lat - half],
            [lon + half, lat + half], [lon - half, lat + half],
            [lon - half, lat - half],
        ]]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": polygon},
            "properties": {
                "cell_id": row.cell_id,
                "hazard": hazard_model.hazard,
                "severity": severity,
                "confidence": round(float(conf), 3),
            },
        })
    return {"type": "FeatureCollection", "features": features}


def export_compact_zones(hazard_model: HazardModel, df_year: pd.DataFrame) -> list:
    """Lightweight [lat, lon, severity, confidence] rows instead of full
    GeoJSON polygons — for bundling many (state, hazard, year) results into
    one payload for a frontend demo. The frontend reconstructs each cell's
    square from `lat`/`lon` + the known grid resolution."""
    X = df_year[hazard_model.feature_cols]
    pred_idx = hazard_model.model.predict(X)
    proba = hazard_model.model.predict_proba(X)
    severities = hazard_model.label_encoder.inverse_transform(pred_idx)
    confidences = proba.max(axis=1)
    return [
        [round(float(row.lat), 2), round(float(row.lon), 2), sev, round(float(conf), 2)]
        for (_, row), sev, conf in zip(df_year.iterrows(), severities, confidences)
    ]


# =============================================================================
# 6. DEMO RUN — trains on Assam floods, exports a sample GeoJSON for the UI
# =============================================================================

if __name__ == "__main__":
    STATE, HAZARD, DEMO_YEAR = "Assam", "flood", 2024

    print(f"Generating synthetic 1925-2025 history for {STATE}...")
    history = generate_synthetic_history(STATE, 1925, 2025)

    print(f"Engineering features for hazard={HAZARD}...")
    history = engineer_features(history, HAZARD)

    print(f"Training {HAZARD} model...")
    hazard_model = train_hazard_model(history, HAZARD, test_year_cutoff=2015)

    print(f"\nGenerating zone map for {STATE}, {DEMO_YEAR}...")
    df_year = history[history.year == DEMO_YEAR]
    geojson = predict_zones(hazard_model, df_year, state=STATE)

    out_path = f"{STATE.lower()}_{HAZARD}_{DEMO_YEAR}_zones.geojson"
    with open(out_path, "w") as f:
        json.dump(geojson, f)
    print(f"Wrote {len(geojson['features'])} zone cells -> {out_path}")

    severity_counts = pd.Series([f["properties"]["severity"] for f in geojson["features"]]).value_counts()
    print("\nSeverity distribution for this year:")
    print(severity_counts)
