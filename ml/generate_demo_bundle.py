"""
LEGACY / OFFLINE-DEMO PATH ONLY -- read this before using it.
================================================================
The frontend's primary data path is now the backend API
(`backend/server.py`), which trains lazily and answers ANY year from
1925 to the current year on demand -- that's what fixed the "only
2000/2010/2018/2024 show up" limitation this script originally had.

This script still exists for one reason: if you want to hand someone
the `frontend/` folder with NO backend running at all (e.g. a static
GitHub Pages demo), you need *some* precomputed data file. Because a
flat JSON file covering every year for every state would be tens of
MB+, DEMO_YEARS below is deliberately still a small sample of years,
not the full range -- for the full range, run the backend instead.

Trains a model for every (state, hazard) pair across all 37 Indian
states/UTs and exports a compact JSON bundle covering the sample years
below, for a zero-backend offline demo of the frontend UI.
"""
import json
import time
from disaster_ml_pipeline import (
    STATE_HAZARDS, generate_synthetic_history,
    engineer_features, train_hazard_model, export_compact_zones,
)
import state_geo

# Sample years only -- see module docstring. Run the backend for full
# 1925-present coverage instead of expanding this list.
DEMO_YEARS = [1950, 1975, 2000, 2010, 2018, 2024]


def main():
    real_states = state_geo.list_states()
    bundle = {"resolutions": {}, "data": {}}
    t_start = time.time()

    for si, state in enumerate(real_states, 1):
        hazards = STATE_HAZARDS.get(state)
        if not hazards:
            continue
        t0 = time.time()
        history = generate_synthetic_history(state, 1925, 2025)
        bundle["resolutions"][state] = state_geo.pick_resolution(state)

        for hazard in hazards:
            featured = engineer_features(history.copy(), hazard)
            model = train_hazard_model(featured, hazard, test_year_cutoff=2015, verbose=False)
            for year in DEMO_YEARS:
                df_year = featured[featured.year == year]
                bundle["data"][f"{state}|{hazard}|{year}"] = export_compact_zones(model, df_year)

        print(f"[{si}/{len(real_states)}] {state} ({', '.join(hazards)}) done in {time.time()-t0:.1f}s")

    with open("demo_bundle.json", "w") as f:
        json.dump(bundle, f, separators=(",", ":"))

    total_predictions = sum(len(v) for v in bundle["data"].values())
    print(f"\nWrote demo_bundle.json: {len(bundle['data'])} (state,hazard,year) combos, "
          f"{total_predictions} total cell predictions, {time.time()-t_start:.1f}s total")


if __name__ == "__main__":
    main()
