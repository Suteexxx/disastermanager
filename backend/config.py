"""
Central config for the live-data layer. Everything secret comes from
environment variables (or a local `.env` — see `.env.example`), never
hardcoded, so this file is safe to commit.
"""
import os

# --- API keys -----------------------------------------------------------
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# --- Crawl behaviour ------------------------------------------------------
CRAWL_INTERVAL_MINUTES = int(os.environ.get("CRAWL_INTERVAL_MINUTES", "60"))
MAX_SIGNAL_AGE_HOURS = int(os.environ.get("MAX_SIGNAL_AGE_HOURS", "48"))
REQUEST_TIMEOUT_SECONDS = 45

# --- Storage --------------------------------------------------------------
DB_PATH = os.environ.get("RAKSHA_DB_PATH", os.path.join(os.path.dirname(__file__), "live_data.sqlite3"))

# --- Sources to crawl -------------------------------------------------------
# Each entry: which state(s)/hazard(s) it's relevant to (used to route the
# extracted signal), a human label, and the URL Firecrawl should scrape.
# "all" means the page is national in scope and gets checked against every
# state that has that hazard.
#
# NOTE: these are the *kind* of official pages this pipeline is designed to
# read (IMD/NDMA/CWC bulletins are public, text-heavy, and update on a
# rolling basis, which is exactly what Firecrawl's markdown-extraction
# scrape endpoint is good at). Swap/add exact URLs as needed — Firecrawl
# handles JS-rendered pages too, so state DMA portals with dynamic content
# work the same way.
SOURCES = [
    {
        "label": "IMD - National Weather Warnings",
        "url": "https://mausam.imd.gov.in/imd_latest/contents/all_india_forecast_bulletin.php",
        "hazards": ["flood", "cyclone", "sandstorm"],
        "states": "all",
    },
    {
        "label": "IMD - Cyclone Warnings",
        "url": "https://mausam.imd.gov.in/imd_latest/contents/cyclone.php",
        "hazards": ["cyclone"],
        "states": "all",
    },
    {
        "label": "CWC - Flood Forecast Monitor",
        "url": "https://ffs.india-water.gov.in/",
        "hazards": ["flood"],
        "states": "all",
    },
    {
        "label": "NDMA - Current Alerts",
        "url": "https://ndma.gov.in/",
        "hazards": ["flood", "landslide", "avalanche", "sandstorm", "cyclone"],
        "states": "all",
    },
]
