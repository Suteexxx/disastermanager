"""
Thin wrapper around the Firecrawl API (https://docs.firecrawl.dev).

Uses the /v1/scrape endpoint to turn a government bulletin page into
clean markdown (Firecrawl handles JS-rendered pages, boilerplate
stripping, etc., which a raw `requests.get()` + BeautifulSoup pass does
not do reliably for these portals).

Fails soft, on purpose: if FIRECRAWL_API_KEY isn't set, or the request
errors out (site down, rate-limited, network blocked), this returns
None instead of raising. The rest of the pipeline is written to treat
"no live data yet" as a normal, expected state -- the ML model's own
prediction is always a complete answer on its own; live data only ever
*adjusts* it on top.
"""
from __future__ import annotations
import requests
import config

FIRECRAWL_SCRAPE_URL = "https://api.firecrawl.dev/v1/scrape"


def is_configured() -> bool:
    return bool(config.FIRECRAWL_API_KEY)


def scrape_url(url: str) -> dict | None:
    """Returns {"markdown": str, "title": str, "url": str} or None."""
    if not is_configured():
        return None
    try:
        resp = requests.post(
            FIRECRAWL_SCRAPE_URL,
            headers={
                "Authorization": f"Bearer {config.FIRECRAWL_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "url": url,
                "formats": ["markdown"],
                "onlyMainContent": True,
                "timeout": config.REQUEST_TIMEOUT_SECONDS * 1000,
            },
            timeout=config.REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data", payload)  # v1 wraps in "data"; be tolerant of shape changes
        markdown = data.get("markdown", "")
        if not markdown:
            return None
        return {
            "markdown": markdown[:20000],  # cap payload size fed to the LLM extractor
            "title": (data.get("metadata") or {}).get("title", url),
            "url": url,
        }
    except requests.RequestException as e:
        print(f"[firecrawl_client] scrape failed for {url}: {e}")
        return None
    except (ValueError, KeyError) as e:
        print(f"[firecrawl_client] unexpected response shape for {url}: {e}")
        return None
