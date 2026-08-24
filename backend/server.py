"""
Raksha Grid backend API.

Replaces the old static `demo_bundle.json` data path. Serves:
  - zone predictions for ANY year from 1925 to the current year, for any
    modeled (state, hazard) -- trained lazily and cached (model_cache.py)
  - a 6-month forward heuristic forecast (ml/forecast.py), optionally
    nudged by live Firecrawl-derived signals
  - live-data status + a manual "refresh now" trigger

Run:  uvicorn server:app --reload --port 8001
(see backend/README section in the project root README.md)
"""
from __future__ import annotations
import sys, os, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import config
import model_cache
import scheduler
import rag_store
import live_feature_bridge
from forecast import forecast_next_months  # ml/forecast.py

app = FastAPI(title="Raksha Grid API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local student project: any localhost:PORT frontend can call this
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    rag_store.init()
    scheduler.start()
    # kick off first crawl + first model warm-up in background threads so
    # server startup itself is instant
    threading.Thread(target=scheduler.run_crawl_cycle, daemon=True).start()
    threading.Thread(target=model_cache.warm_cache_all, daemon=True).start()


@app.on_event("shutdown")
def _shutdown():
    scheduler.stop()


@app.get("/api/states")
def get_states():
    return {"states": model_cache.list_states()}


@app.get("/api/hazards")
def get_hazards(state: str):
    hazards = model_cache.hazards_for_state(state)
    if not hazards:
        raise HTTPException(404, f"Unknown or unmodeled state: {state}")
    return {"state": state, "hazards": hazards}


@app.get("/api/year_range")
def get_year_range(state: str | None = None):
    return model_cache.year_range(state)


@app.get("/api/zones")
def get_zones(state: str, hazard: str, year: int):
    yr = model_cache.year_range(state)
    if not (yr["min_year"] <= year <= yr["max_year"]):
        raise HTTPException(400, f"year must be between {yr['min_year']} and {yr['max_year']}")
    try:
        model, engineered = model_cache.get_or_train(state, hazard)
    except ValueError as e:
        raise HTTPException(400, str(e))

    df_year = engineered[engineered.year == year]
    if df_year.empty:
        raise HTTPException(404, f"No data for {state}/{hazard}/{year}")

    live_meta = None
    if year == yr["current_year"]:
        df_year, live_meta = live_feature_bridge.adjust_features_for_year(df_year, hazard, state)

    from disaster_ml_pipeline import export_compact_zones
    cells = export_compact_zones(model, df_year)
    return {
        "state": state, "hazard": hazard, "year": year,
        "resolution": __import__("state_geo").pick_resolution(state),
        "cells": cells,
        "live": live_meta,
    }


@app.get("/api/forecast")
def get_forecast(state: str, hazard: str, months: int = 6):
    months = max(1, min(months, 12))
    try:
        model, engineered = model_cache.get_or_train(state, hazard)
    except ValueError as e:
        raise HTTPException(400, str(e))

    yr = model_cache.year_range(state)
    df_latest = engineered[engineered.year == yr["current_year"]]
    if df_latest.empty:
        df_latest = engineered[engineered.year == engineered.year.max()]

    live_fn = live_feature_bridge.make_live_multiplier_fn(state)
    months_out = forecast_next_months(model, df_latest, hazard, n_months=months, live_multiplier_fn=live_fn)
    return {
        "state": state, "hazard": hazard,
        "resolution": __import__("state_geo").pick_resolution(state),
        "months": months_out,
    }


@app.get("/api/live/status")
def live_status():
    return {
        "db": rag_store.status(),
        "last_crawl": scheduler.last_run_summary(),
        "sources": [{"label": s["label"], "url": s["url"]} for s in config.SOURCES],
        "crawl_interval_minutes": config.CRAWL_INTERVAL_MINUTES,
    }


@app.post("/api/live/refresh")
def live_refresh():
    result = scheduler.run_crawl_cycle()
    return result


@app.get("/api/live/context")
def live_context(state: str, hazard: str, k: int = 3):
    return {"context": rag_store.retrieve_context(state, hazard, k)}


@app.get("/")
def root():
    return {
        "name": "Raksha Grid API", "status": "ok",
        "docs": "/docs", "states": len(model_cache.list_states()),
    }
