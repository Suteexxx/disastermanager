"""
Lightweight local store for crawled documents + extracted signals.

Deliberately NOT a vector database -- for this project's scale (a
handful of government bulletin pages, refreshed hourly) a full embedding
+ vector-DB stack is overkill infra for what's actually a few dozen
documents at any time. SQLite (stdlib, zero extra install) holds the
raw documents; retrieval for the "RAG" side (e.g. if you want to show
"why was this cell adjusted" text in the UI) uses TF-IDF similarity
(scikit-learn, already a project dependency) over the stored documents.
The one thing that actually matters for the ML pipeline -- the
structured numeric signal -- is queried directly by state+hazard, not
by similarity search, because that lookup needs to be exact.

If this needs to scale to hundreds of sources later, swap this module
for a real vector store (e.g. Chroma) -- `latest_signal()` and
`retrieve_context()` are the two functions everything else calls, so
that's the only surface to keep stable.
"""
from __future__ import annotations
import sqlite3
import time
from contextlib import contextmanager
from typing import Optional
import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT, title TEXT, raw_markdown TEXT, fetched_at REAL
);
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER, state TEXT, hazard TEXT,
    magnitude REAL, confidence REAL, summary TEXT,
    source_url TEXT, fetched_at REAL
);
CREATE INDEX IF NOT EXISTS idx_signals_state_hazard ON signals(state, hazard, fetched_at);
"""


@contextmanager
def _conn():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init():
    with _conn() as c:
        c.executescript(_SCHEMA)


def save_document(url: str, title: str, markdown: str) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO documents (url, title, raw_markdown, fetched_at) VALUES (?,?,?,?)",
            (url, title, markdown, time.time()),
        )
        return cur.lastrowid


def save_signals(doc_id: int, source_url: str, signals: list[dict]):
    with _conn() as c:
        now = time.time()
        c.executemany(
            "INSERT INTO signals (doc_id, state, hazard, magnitude, confidence, summary, source_url, fetched_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            [(doc_id, s["state"], s["hazard"], s["magnitude"], s["confidence"], s["summary"], source_url, now)
             for s in signals],
        )


def latest_signal(state: str, hazard: str, max_age_hours: Optional[int] = None) -> Optional[dict]:
    max_age_hours = max_age_hours or config.MAX_SIGNAL_AGE_HOURS
    cutoff = time.time() - max_age_hours * 3600
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM signals WHERE state=? AND hazard=? AND fetched_at>=? "
            "ORDER BY fetched_at DESC LIMIT 1",
            (state, hazard, cutoff),
        ).fetchone()
        return dict(row) if row else None


def all_recent_signals(max_age_hours: Optional[int] = None) -> list[dict]:
    max_age_hours = max_age_hours or config.MAX_SIGNAL_AGE_HOURS
    cutoff = time.time() - max_age_hours * 3600
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM signals WHERE fetched_at>=? ORDER BY fetched_at DESC", (cutoff,)
        ).fetchall()
        return [dict(r) for r in rows]


def status() -> dict:
    with _conn() as c:
        doc_count = c.execute("SELECT COUNT(*) n FROM documents").fetchone()["n"]
        last_doc = c.execute("SELECT MAX(fetched_at) t FROM documents").fetchone()["t"]
        sig_count = c.execute(
            "SELECT COUNT(*) n FROM signals WHERE fetched_at>=?",
            (time.time() - config.MAX_SIGNAL_AGE_HOURS * 3600,),
        ).fetchone()["n"]
        return {
            "documents_crawled_total": doc_count,
            "last_crawl_epoch": last_doc,
            "active_signals": sig_count,
            "max_signal_age_hours": config.MAX_SIGNAL_AGE_HOURS,
        }


def retrieve_context(state: str, hazard: str, k: int = 3) -> list[dict]:
    """Simple TF-IDF similarity over recent documents, scoped to ones that
    produced a signal for this state+hazard -- used for an optional
    'why was this adjusted' explanation panel in the UI."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    with _conn() as c:
        rows = c.execute(
            "SELECT DISTINCT d.id, d.title, d.url, d.raw_markdown FROM documents d "
            "JOIN signals s ON s.doc_id = d.id WHERE s.state=? AND s.hazard=? "
            "ORDER BY d.fetched_at DESC LIMIT 20",
            (state, hazard),
        ).fetchall()
    if not rows:
        return []
    texts = [r["raw_markdown"] for r in rows]
    query = f"{state} {hazard} risk alert"
    try:
        vec = TfidfVectorizer(stop_words="english", max_features=2000).fit(texts + [query])
        mat = vec.transform(texts)
        qv = vec.transform([query])
        sims = (mat @ qv.T).toarray().ravel()
    except ValueError:
        sims = [0] * len(rows)
    ranked = sorted(zip(rows, sims), key=lambda x: -x[1])[:k]
    return [{"title": r["title"], "url": r["url"], "snippet": r["raw_markdown"][:280]} for r, _ in ranked]
