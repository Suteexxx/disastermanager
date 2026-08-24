# Raksha Grid — All-India Hazard Zone Predictor (Live Data Edition)

Trains a per-hazard ML model for every Indian state/UT (37 total) using
each state's REAL boundary shape, serves predictions for **any year from
1925 to today** plus a **6-month forward forecast**, and can nudge both
using **live signals scraped from official bulletins via Firecrawl**.
Visualized on an interactive Leaflet map that follows the actual
coastline/border of whichever state you select.

## What changed in this version

1. **Backend API instead of a static JSON bundle.** The old
   `frontend/demo_bundle.json` only ever had 4 hardcoded years
   (2000/2010/2018/2024) because pre-computing every year for every state
   as flat JSON doesn't scale. The new `backend/server.py` trains each
   (state, hazard) model once and caches it in memory, then answers
   **any** year 1925–present on demand by re-slicing that model's own
   history and re-running prediction — a few hundred milliseconds per
   request. This is what actually fixes "I can't see 2011/2012/2013" —
   those years were never missing from the model, only from the static
   file.
2. **6-month forward forecast** (`ml/forecast.py`) — a documented seasonal
   heuristic layered on top of the trained yearly model (see that file's
   docstring for exactly what it is and isn't; it is **not** a second
   trained model).
3. **Firecrawl-powered live data feed** (`backend/`) — a scheduled job
   scrapes a configurable list of official sources (IMD, CWC, NDMA, ...),
   an LLM extraction step (Groq) turns the scraped text into small
   structured signals per state+hazard, and those signals nudge the
   *current year* and the *forecast* — never the historical record, which
   stays exactly what the trained model says.
4. **The core ML pipeline logic is unchanged** — `disaster_ml_pipeline.py`
   is untouched. Live data only ever adjusts specific feature columns
   right before prediction (`backend/live_feature_bridge.py`); training,
   zoning, and the frontend's visual design are the same as before.

## Project layout

```
raksha_grid_package/
├── ml/
│   ├── disaster_ml_pipeline.py   # core pipeline: data, features, model, zoning (UNCHANGED)
│   ├── forecast.py               # NEW: 6-month seasonal forecast layer
│   ├── state_geo.py              # loads real state polygons, builds in-shape grids
│   ├── generate_demo_bundle.py   # LEGACY: static offline-demo export (small year sample only)
│   └── india_states.geojson      # real state/UT boundaries
├── backend/                      # NEW: live API + Firecrawl/RAG layer
│   ├── server.py                 # FastAPI app — the frontend's data source now
│   ├── model_cache.py            # lazy train-and-cache per (state, hazard)
│   ├── firecrawl_client.py       # Firecrawl /v1/scrape wrapper (fails soft, no key = skipped)
│   ├── llm_extract.py            # Groq LLM extraction -> structured signals (keyword fallback if no key)
│   ├── rag_store.py              # SQLite store for scraped docs + signals, TF-IDF retrieval
│   ├── live_feature_bridge.py    # signal -> feature-column adjustment bridge
│   ├── scheduler.py              # APScheduler background crawl loop + manual trigger
│   ├── config.py                 # source list, intervals, env-var loading
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    ├── index.html                # map UI shell (+ live panel, mode toggle, forecast slider)
    ├── app.js                    # calls the backend API instead of a static bundle
    ├── config.js                 # NEW: one-line API_BASE setting
    ├── style.css                 # dark theme (+ small additions)
    ├── demo_bundle.json          # LEGACY, no longer loaded by app.js — see generate_demo_bundle.py
    ├── india_states.geojson
    └── lib/                      # Leaflet, bundled locally
```

## 1. Run the backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# optional but recommended — enables live crawling; works fine without it too
cp .env.example .env
# edit .env: FIRECRAWL_API_KEY=..., GROQ_API_KEY=...

uvicorn server:app --reload --port 8001
```

Open `http://127.0.0.1:8001/docs` to see/try every endpoint directly.

**No API keys?** The backend still runs and still answers every year
1925–present from the trained model — you just won't get live-adjusted
current-year zones or forecasts. `firecrawl_client.py` and `llm_extract.py`
both fail soft (return nothing) instead of crashing when a key is missing,
and `llm_extract.py` has a keyword-based fallback extractor so the whole
crawl → extract → adjust pipeline still runs end-to-end for demo purposes.

## 2. Run the frontend

```bash
cd frontend
python3 -m http.server 8000
# open http://localhost:8000
```

`frontend/config.js` points at `http://127.0.0.1:8001` by default — change
`API_BASE` there if your backend runs elsewhere. If the backend isn't
reachable, the sidebar shows a clear warning instead of failing silently.

## 3. Using it

- **Historical** mode: year slider now covers 1925 → the current year for
  every state/hazard, not just 4 sample years.
- **6-Month Forecast** mode: month slider steps through the next 6 months
  from today. A `FORECAST` badge marks the seasonal-heuristic basis; a
  `LIVE` badge means a fresh scraped signal is also nudging that view.
- **Live Data Feed card**: shows last crawl time, documents crawled, and
  active signals; "Refresh now" triggers an immediate crawl cycle instead
  of waiting for the next scheduled interval (`CRAWL_INTERVAL_MINUTES` in
  `.env`, default 60).

## Notes (same caveats as before, still true)

- **The training data is synthetic** — see the module docstring at the
  top of `ml/disaster_ml_pipeline.py` for exactly which real data sources
  (IMD, CWC, NCS, DGRE, Forest Survey of India, EM-DAT) each synthetic
  column stands in for. Swapping in real data means replacing
  `generate_synthetic_history()` with real ETL jobs that populate the
  same column names — nothing downstream (features → train → zone →
  live-adjust → forecast) needs to change.
- **Base map tiles** load live from OpenStreetMap and need internet.
- **`backend/config.py`'s `SOURCES` list** is a starting point, not a
  finished crawl config — add/replace URLs for the state DMA portals or
  IMD regional bulletins most relevant to your use case. Firecrawl
  handles JS-rendered pages the same way, so dynamic portals work too.
