"""
Background job that runs the crawl -> extract -> store cycle on a fixed
interval (config.CRAWL_INTERVAL_MINUTES), plus a manual trigger for the
"Refresh now" button in the UI. Uses APScheduler's BackgroundScheduler so
it runs inside the same FastAPI process -- no separate worker/queue
needed at this scale (a handful of sources, hourly).
"""
from __future__ import annotations
import time
from apscheduler.schedulers.background import BackgroundScheduler

import config
import firecrawl_client
import llm_extract
import rag_store

_scheduler: BackgroundScheduler | None = None
_last_run_summary: dict = {"status": "not_run_yet"}


def run_crawl_cycle() -> dict:
    """One full pass over config.SOURCES. Safe to call even with no API
    keys configured -- see firecrawl_client / llm_extract fallback docs."""
    global _last_run_summary
    t0 = time.time()
    results = []
    for source in config.SOURCES:
        scraped = firecrawl_client.scrape_url(source["url"])
        if scraped is None:
            results.append({"label": source["label"], "url": source["url"], "status": "skipped_or_failed"})
            continue
        signals_raw = llm_extract.extract_signals(scraped["markdown"], source["url"])
        # keep only signals for hazards this source is actually tagged for
        signals = [s for s in signals_raw if s["hazard"] in source["hazards"]]
        doc_id = rag_store.save_document(scraped["url"], scraped["title"], scraped["markdown"])
        if signals:
            rag_store.save_signals(doc_id, source["url"], signals)
        results.append({
            "label": source["label"], "url": source["url"], "status": "ok",
            "signals_extracted": len(signals),
        })

    _last_run_summary = {
        "status": "completed",
        "duration_sec": round(time.time() - t0, 1),
        "sources": results,
        "firecrawl_configured": firecrawl_client.is_configured(),
        "groq_configured": llm_extract.is_configured(),
    }
    print(f"[scheduler] crawl cycle done in {_last_run_summary['duration_sec']}s: {_last_run_summary}")
    return _last_run_summary


def last_run_summary() -> dict:
    return _last_run_summary


def start():
    global _scheduler
    rag_store.init()
    if _scheduler is not None:
        return
    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        run_crawl_cycle, "interval",
        minutes=config.CRAWL_INTERVAL_MINUTES,
        id="raksha_grid_crawl", next_run_time=None,  # first run kicked off explicitly by server.py startup
    )
    _scheduler.start()


def stop():
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
