"""
Turns raw scraped bulletin text into a small set of STRUCTURED numeric
signals the ML pipeline can actually use. A HistGradientBoostingClassifier
takes numbers (rainfall_anomaly_pct, wind_speed_kmh, ...), not paragraphs
-- so this is the bridge step of the "RAG pipeline": retrieve (Firecrawl
scrape) -> ground the LLM call in that retrieved text -> extract structured
signal -> the signal is what actually touches the model's features.

Primary path: Groq (llama-3.3-70b-versatile, matches the rest of this
project's LLM usage) does the extraction, forced into strict JSON.

Fallback path (no GROQ_API_KEY set): a small keyword/regex scan. It is
deliberately conservative -- it only ever nudges risk up when it sees an
explicit alert-type keyword near a state/hazard name, and it always
reports low confidence. This keeps the whole live-data feature usable
end-to-end (crawl -> extract -> adjust -> forecast) without any paid API
key, for demoing or grading, while the Groq path is what you'd actually
run in production for a real signal.
"""
from __future__ import annotations
import json
import re
import requests
import config

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

EXTRACTION_SYSTEM_PROMPT = """You are a structured-data extractor for an Indian disaster-risk pipeline.
Given raw text scraped from an official bulletin (IMD/NDMA/CWC/state DMA), extract at most 6 signals.
Each signal says: for a given Indian state and hazard, is this bulletin describing conditions that should
push modeled risk UP, DOWN, or leave it unchanged, and by how much.

Respond with ONLY a JSON array (no prose, no markdown fences), where each element is:
{
  "state": "<Indian state/UT name, must match official names, e.g. 'Assam', 'Odisha'>",
  "hazard": "<one of: flood, landslide, avalanche, sandstorm, cyclone>",
  "magnitude": <float from -1.0 to 1.0, 0 = no change, 1.0 = severe active alert, -0.3 = conditions easing>,
  "confidence": <float 0.0-1.0, how directly the text supports this>,
  "summary": "<one short sentence, plain English, max 20 words>"
}
If the text contains nothing relevant to any state+hazard, respond with an empty JSON array: []
"""


def is_configured() -> bool:
    return bool(config.GROQ_API_KEY)


def extract_signals(markdown_text: str, source_url: str) -> list[dict]:
    if is_configured():
        try:
            return _extract_with_groq(markdown_text)
        except Exception as e:
            print(f"[llm_extract] Groq extraction failed, falling back to keyword scan: {e}")
    return _extract_with_keywords(markdown_text)


def _extract_with_groq(markdown_text: str) -> list[dict]:
    resp = requests.post(
        GROQ_CHAT_URL,
        headers={"Authorization": f"Bearer {config.GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": config.GROQ_MODEL,
            "temperature": 0,
            "max_tokens": 1200,
            "messages": [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": markdown_text[:12000]},
            ],
        },
        timeout=config.REQUEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()
    content = re.sub(r"^```(json)?|```$", "", content.strip(), flags=re.MULTILINE).strip()
    signals = json.loads(content)
    out = []
    for s in signals:
        if not isinstance(s, dict) or "state" not in s or "hazard" not in s:
            continue
        out.append({
            "state": str(s.get("state", "")).strip(),
            "hazard": str(s.get("hazard", "")).strip().lower(),
            "magnitude": float(max(-1.0, min(1.0, s.get("magnitude", 0)))),
            "confidence": float(max(0.0, min(1.0, s.get("confidence", 0.3)))),
            "summary": str(s.get("summary", ""))[:200],
        })
    return out


# --- fallback: no API key needed -------------------------------------------

_ALERT_WORDS = {
    "severe": 0.8, "red alert": 0.9, "orange alert": 0.6, "yellow alert": 0.3,
    "warning": 0.5, "heavy rainfall": 0.5, "very heavy rainfall": 0.7,
    "extremely heavy": 0.85, "landfall": 0.7, "depression": 0.4,
    "cyclonic storm": 0.6, "flood alert": 0.6, "evacuat": 0.7,
}
_HAZARD_WORDS = {
    "flood": ["flood", "river", "inundat", "overflow"],
    "cyclone": ["cyclone", "cyclonic", "storm surge", "depression"],
    "landslide": ["landslide", "slope failure", "mudslide"],
    "avalanche": ["avalanche", "snow slide"],
    "sandstorm": ["sandstorm", "dust storm", "duststorm"],
}
_STATE_NAMES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal", "Jammu and Kashmir", "Ladakh", "Delhi", "Puducherry",
    "Chandigarh", "Andaman and Nicobar Islands", "Lakshadweep",
]


def _extract_with_keywords(text: str) -> list[dict]:
    low = text.lower()
    out = []
    for state in _STATE_NAMES:
        if state.lower() not in low:
            continue
        # look within a window around the state mention for hazard + alert words
        idx = low.find(state.lower())
        window = low[max(0, idx - 300): idx + 300]
        for hazard, words in _HAZARD_WORDS.items():
            if not any(w in window for w in words):
                continue
            magnitude = max((v for k, v in _ALERT_WORDS.items() if k in window), default=0.0)
            if magnitude <= 0:
                continue
            out.append({
                "state": state, "hazard": hazard, "magnitude": round(magnitude, 2),
                "confidence": 0.25,  # always low-confidence, this is a blunt fallback
                "summary": f"Keyword match near '{state}' suggests active {hazard} conditions (no LLM key set).",
            })
    return out[:6]
