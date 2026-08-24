"""
The actual bridge between "live RAG signal" and "ML pipeline feature".

A signal from llm_extract.py is a single scalar `magnitude` (-1..1) per
state+hazard. This module turns that into per-hazard adjustments on the
SAME feature columns disaster_ml_pipeline.py already trains on -- so the
existing trained model can score the adjusted row directly, with zero
changes to the model or its feature list. This is the "keep the logic
the same" requirement: nothing about training or zoning changes, only
one extra step is inserted before prediction.

Two entry points:
  - adjust_features_for_year(df_year, hazard, state) -> used by /api/zones
    for "current year" requests: overrides the live-relevant columns on
    every cell for that state+hazard if a fresh signal exists.
  - live_multiplier_fn(hazard, lat, lon) factory -> used by ml/forecast.py
    to nudge the 6-month seasonal forecast up/down using the same signal.
"""
from __future__ import annotations
from typing import Optional
import rag_store

# Which raw columns each hazard's magnitude should push, and in which
# direction/scale. Kept intentionally simple (linear nudge) so the effect
# stays inspectable and doesn't fight the model's own learned decision
# boundaries -- a magnitude of 1.0 (max severity alert) shifts these
# columns by roughly one "bad" standard deviation, not by extreme jumps.
_HAZARD_COLUMN_NUDGES = {
    "flood": {"rainfall_7day_mm": 120, "rainfall_anomaly_pct": 40, "soil_saturation_index": 0.25},
    "landslide": {"rainfall_24hr_mm": 50, "rainfall_antecedent_mm": 80, "seismic_activity_index": 0.15},
    "avalanche": {"snowpack_depth_cm": 60, "new_snow_24hr_cm": 15, "temperature_swing_c": 5},
    "sandstorm": {"wind_speed_kmh": 25, "aridity_index": 0.2, "land_degradation_index": 0.15},
    "cyclone": {"wind_speed_kmh": 60, "central_pressure_hpa": -25, "historical_landfall_freq": 0.2},
}


def get_signal(state: str, hazard: str) -> Optional[dict]:
    return rag_store.latest_signal(state, hazard)


def adjust_features_for_year(df_year, hazard: str, state: str):
    """Returns (adjusted_df, meta). meta is None if no live signal applied."""
    signal = get_signal(state, hazard)
    if not signal or signal["magnitude"] == 0:
        return df_year, None

    df = df_year.copy()
    magnitude = signal["magnitude"]
    for col, scale in _HAZARD_COLUMN_NUDGES.get(hazard, {}).items():
        if col in df.columns:
            df[col] = df[col] + magnitude * scale

    meta = {
        "applied": True,
        "magnitude": magnitude,
        "confidence": signal["confidence"],
        "summary": signal["summary"],
        "source_url": signal["source_url"],
    }
    return df, meta


def make_live_multiplier_fn(state: str):
    """Factory for ml/forecast.py: returns a function(hazard, lat, lon) ->
    (multiplier, is_live). Signal is per state+hazard (not per-cell -- the
    live sources this project targets are state/region-level bulletins,
    not grid-cell-resolution), so every cell in the state gets the same
    multiplier for a given hazard."""
    cache: dict[str, tuple[float, bool]] = {}

    def _fn(hazard: str, lat: float, lon: float):
        if hazard not in cache:
            signal = get_signal(state, hazard)
            if signal and signal["magnitude"] != 0:
                # magnitude in [-1,1] -> multiplier in [0.5, 1.8], centered at 1.0
                mult = 1.0 + signal["magnitude"] * 0.8
                cache[hazard] = (max(0.4, mult), True)
            else:
                cache[hazard] = (1.0, False)
        return cache[hazard]

    return _fn
