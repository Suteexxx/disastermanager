"""
Lazy train-and-cache layer over ml/disaster_ml_pipeline.py.

Why lazy instead of pre-baking a static bundle (the old demo_bundle.json
approach): that file only ever contained 4 hardcoded years (2000, 2010,
2018, 2024) because storing every (state, hazard, year) combo as flat
JSON for the full 1925-present range would be tens of millions of rows.
Training is fast per state (a few seconds), and inference on an already-
trained model for one extra year is near-instant -- so instead of
pre-computing everything, we train once per (state, hazard) on first
request and keep the model + full history DataFrame in memory. Any year
from 1925 to the current year then just re-slices that in-memory history
and re-runs prediction, which is what fixes the "only specific years
show up" issue directly, rather than papering over it with a bigger
hardcoded list.
"""
from __future__ import annotations
import datetime
import sys
import os
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml"))
import disaster_ml_pipeline as pipeline  # noqa: E402
import state_geo  # noqa: E402

CURRENT_YEAR = datetime.date.today().year
MIN_YEAR = 1925

_lock = threading.Lock()
_history_cache: dict[str, "pd.DataFrame"] = {}      # state -> raw synthetic history (all hazards)
_model_cache: dict[tuple[str, str], pipeline.HazardModel] = {}  # (state, hazard) -> trained model
_engineered_cache: dict[tuple[str, str], "pd.DataFrame"] = {}   # (state, hazard) -> engineered df


def list_states() -> list[str]:
    return [s for s in state_geo.list_states() if s in pipeline.STATE_HAZARDS]


def hazards_for_state(state: str) -> list[str]:
    return pipeline.STATE_HAZARDS.get(state, [])


def _get_history(state: str):
    if state not in _history_cache:
        _history_cache[state] = pipeline.generate_synthetic_history(state, MIN_YEAR, CURRENT_YEAR)
    return _history_cache[state]


def get_or_train(state: str, hazard: str) -> tuple[pipeline.HazardModel, "pd.DataFrame"]:
    """Returns (trained model, engineered full-history df) for (state, hazard),
    training + caching on first call."""
    key = (state, hazard)
    if key in _model_cache:
        return _model_cache[key], _engineered_cache[key]

    with _lock:
        if key in _model_cache:  # re-check after acquiring lock
            return _model_cache[key], _engineered_cache[key]
        if hazard not in pipeline.STATE_HAZARDS.get(state, []):
            raise ValueError(f"{hazard} is not a modeled hazard for {state}")

        history = _get_history(state)
        engineered = pipeline.engineer_features(history.copy(), hazard)
        model = pipeline.train_hazard_model(
            engineered, hazard, test_year_cutoff=min(2015, CURRENT_YEAR - 1), verbose=False
        )
        _model_cache[key] = model
        _engineered_cache[key] = engineered
        return model, engineered


def year_range(state: str) -> dict:
    return {"min_year": MIN_YEAR, "max_year": CURRENT_YEAR, "current_year": CURRENT_YEAR}


def warm_cache_all():
    """Optional: called from a background thread at startup to pre-train
    every (state, hazard) pair so the first user request for ANY state is
    already fast, instead of paying the ~2-5s training cost on first click.
    Not required -- get_or_train() lazily trains on demand regardless."""
    for state in list_states():
        for hazard in hazards_for_state(state):
            try:
                get_or_train(state, hazard)
            except Exception as e:
                print(f"[model_cache] warm-up failed for {state}/{hazard}: {e}")
