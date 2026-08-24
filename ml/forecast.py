"""
Raksha Grid — 6-Month Forward Hazard Forecast (heuristic seasonal layer)
=========================================================================

IMPORTANT — what this module IS and IS NOT, read before trusting it:

`disaster_ml_pipeline.py` trains one classifier per hazard on YEARLY
cell-year rows. It has never seen a "month" as a feature, so it cannot,
by itself, produce a statistically-trained month-by-month forecast.
Building a real monthly model would need monthly-resolution training
labels (IMD/CWC actually do publish these, e.g. monthly rainfall
departure bulletins) which this project does not yet ingest.

What this module DOES instead, honestly:
  1. Takes the trained yearly HazardModel's prediction for the most
     recent complete year at each cell -> a continuous 0-1 risk index
     (weighted by class probability, not just the argmax bucket).
  2. Multiplies that base risk by a fixed, documented SEASONAL_PROFILE
     per hazard (e.g. flood risk is climatologically concentrated in
     Jun-Sep across India — IMD long-period monsoon normals), to shape
     the next 6 months around India's known hazard seasonality.
  3. Optionally multiplies again by a LIVE signal magnitude coming from
     backend/live_feature_bridge.py (real-time Firecrawl-derived
     adjustment for that state+hazard, e.g. an active IMD flood alert),
     if one exists and is fresh enough.
  4. Confidence is deliberately decayed the further out the month is,
     because a forecast for month+6 is genuinely less certain than one
     for month+1 -- this is NOT a real growing-uncertainty estimate from
     the model itself, it's an explicit, documented penalty so the UI
     doesn't overstate confidence in distant months.

This is a transparent heuristic layer on top of a real trained model —
not a second trained model. Swapping in a genuinely trained monthly
model later means replacing `_seasonal_multiplier()` and the confidence
decay with real outputs; the call signature (`forecast_next_months`)
does not need to change.
"""
from __future__ import annotations
import datetime as _dt
from typing import Dict, List, Optional
import numpy as np

from disaster_ml_pipeline import HazardModel, SEVERITY_LABELS

# Relative seasonal intensity per calendar month (1=Jan ... 12=Dec),
# normalized so each hazard's 12 values average to ~1.0. Grounded in
# IMD/NDMA's well-documented climatological seasonality for India:
#   flood      -> SW monsoon (Jun-Sep) dominant
#   landslide  -> tracks monsoon rainfall (Jun-Sep), small winter bump
#                 in the Himalaya from snowmelt-saturated slopes
#   avalanche  -> peak snowpack + freeze-thaw, Dec-Mar
#   sandstorm  -> pre-monsoon heat + dry westerlies, Apr-Jun
#   cyclone    -> bimodal: pre-monsoon (Apr-Jun) and post-monsoon
#                 (Oct-Dec) — the two Bay of Bengal / Arabian Sea
#                 cyclogenesis windows
SEASONAL_PROFILE: Dict[str, Dict[int, float]] = {
    "flood":      {1:.2,2:.2,3:.3,4:.4,5:.6,6:1.6,7:2.2,8:2.1,9:1.6,10:.7,11:.3,12:.2},
    "landslide":  {1:.3,2:.3,3:.4,4:.5,5:.7,6:1.7,7:2.1,8:2.0,9:1.5,10:.7,11:.4,12:.3},
    "avalanche":  {1:2.1,2:2.0,3:1.6,4:.7,5:.3,6:.1,7:.1,8:.1,9:.2,10:.5,11:1.2,12:1.9},
    "sandstorm":  {1:.4,2:.6,3:1.2,4:2.0,5:2.2,6:1.8,7:.8,8:.5,9:.4,10:.4,11:.3,12:.4},
    "cyclone":    {1:.3,2:.2,3:.3,4:1.3,5:2.0,6:.9,7:.4,8:.3,9:.6,10:1.9,11:2.3,12:1.2},
}

_SEVERITY_INDEX = {"safe": 0.0, "low": 0.33, "moderate": 0.66, "severe": 1.0}
_ORDERED = SEVERITY_LABELS  # ["safe","low","moderate","severe"]


def _seasonal_multiplier(hazard: str, month: int) -> float:
    return SEASONAL_PROFILE.get(hazard, {}).get(month, 1.0)


def _bucket(risk: float) -> str:
    if risk < 0.25:
        return "safe"
    if risk < 0.5:
        return "low"
    if risk < 0.75:
        return "moderate"
    return "severe"


def _base_risk_index(hazard_model: HazardModel, df_latest: "pd.DataFrame") -> np.ndarray:
    """Continuous 0-1 risk score per cell = class-probability-weighted
    severity index, instead of collapsing straight to the argmax bucket.
    Keeps the forecast smooth month-to-month instead of jumping wholesale
    between 4 discrete buckets."""
    X = df_latest[hazard_model.feature_cols]
    proba = hazard_model.model.predict_proba(X)  # columns follow label_encoder.classes_ order
    weights = np.array([_SEVERITY_INDEX[c] for c in hazard_model.label_encoder.classes_])
    return proba @ weights, proba.max(axis=1)


def forecast_next_months(
    hazard_model: HazardModel,
    df_latest_year: "pd.DataFrame",
    hazard: str,
    n_months: int = 6,
    start_date: Optional[_dt.date] = None,
    live_multiplier_fn=None,
) -> List[dict]:
    """
    Returns a list of `n_months` monthly zone bundles:
      [{month, year, month_label, live_adjusted, cells: [[lat,lon,severity,confidence], ...]}, ...]

    `live_multiplier_fn(hazard, lat, lon) -> (multiplier, is_live)` is an
    optional hook (see backend/live_feature_bridge.py) that nudges the
    seasonal risk using a fresh Firecrawl-derived signal for that
    state+hazard, if one exists. Defaults to a no-op (multiplier=1.0).
    """
    if live_multiplier_fn is None:
        live_multiplier_fn = lambda hazard, lat, lon: (1.0, False)

    start_date = start_date or _dt.date.today()
    base_risk, base_conf = _base_risk_index(hazard_model, df_latest_year)
    lats = df_latest_year["lat"].to_numpy()
    lons = df_latest_year["lon"].to_numpy()
    cell_ids = df_latest_year["cell_id"].to_numpy()

    months_out = []
    for step in range(1, n_months + 1):
        target = _add_months(start_date, step)
        seasonal_mult = _seasonal_multiplier(hazard, target.month)
        # confidence decay: explicit, documented, NOT model-derived (see docstring)
        horizon_decay = max(0.35, 1.0 - 0.08 * step)

        cells = []
        any_live = False
        for i in range(len(lats)):
            live_mult, is_live = live_multiplier_fn(hazard, float(lats[i]), float(lons[i]))
            any_live = any_live or is_live
            risk = float(np.clip(base_risk[i] * seasonal_mult * live_mult, 0, 1))
            severity = _bucket(risk)
            confidence = round(float(np.clip(base_conf[i] * horizon_decay, 0.05, 0.99)), 2)
            cells.append([round(float(lats[i]), 2), round(float(lons[i]), 2), severity, confidence])

        months_out.append({
            "month": target.month,
            "year": target.year,
            "month_label": target.strftime("%b %Y"),
            "months_ahead": step,
            "live_adjusted": any_live,
            "cells": cells,
        })
    return months_out


def _add_months(d: _dt.date, n: int) -> _dt.date:
    month = d.month - 1 + n
    year = d.year + month // 12
    month = month % 12 + 1
    return _dt.date(year, month, 1)
