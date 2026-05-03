import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "spotter.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_conn()
    conn.executescript("""
       CREATE TABLE IF NOT EXISTS plates (
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           text TEXT NOT NULL,
           country TEXT,
           confidence REAL,
           valid_format INTEGER,
           source TEXT,
           timestamp TEXT DEFAULT (datetime('now')),
           crop BLOB
       );

       CREATE TABLE IF NOT EXISTS watchlist (
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           text TEXT NOT NULL,
           notes TEXT,
           added_at TEXT DEFAULT (datetime('now'))
       )
   """)
    conn.commit()
    conn.close()


def insert_plate(text, country, confidence, valid_format, source, crop_bytes=None):
    conn = get_conn()
    cur = conn.execute(
        """
        INSERT INTO plates (text, country, confidence, valid_format, source, crop)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        (text, country, confidence, valid_format, source, crop_bytes),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_history(limit=50, offset=0, country=None, source=None):
    conn = get_conn()
    query = "SELECT id, text, country, confidence, valid_format, source, timestamp FROM plates where 1=1"
    params = []

    if country:
        query += " AND country = ?"
        params.append(country)
    if source:
        query += " AND source = ?"
        params.append(source)

    query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats():
    conn = get_conn()

    total = conn.execute("SELECT COUNT(*) FROM plates").fetchone()[0]
    by_country = conn.execute(
        "SELECT country, COUNT(*) as count FROM plates GROUP BY country DESC"
    ).fetchall()
    by_source = conn.execute(
        "SELECT source, COUNT(*) as count FROM plates GROUP BY source ORDER BY count DESC"
    ).fetchall()
    top_plates = conn.execute(
        "SELECT text, country, COUNT(*) as count FROM plates GROUP BY text ORDER BY count DESC LIMIT 10"
    ).fetchall()
    by_hour = conn.execute(
        "SELECT strftime('%H', timestamp) as hour, COUNT(*) as count FROM plates GROUP BY hour ORDER BY hour"
    ).fetchall()
    conn.close()
    return {
        "total": total,
        "by_country": [dict(r) for r in by_country],
        "by_source": [dict(r) for r in by_source],
        "top_plates": [dict(r) for r in top_plates],
        "by_hour": [dict(r) for r in by_hour],
    }


def check_watchlist(text):
    conn = get_conn()
    row = conn.execute("SELECT * FROM watchlist WHERE text = ?", (text,)).fetchone()
    conn.close()
    return dict(row) if row else None


def add_to_watchlist(text, notes=None):
    conn = get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO watchlist (text, notes) VALUES (?, ?)", (text, notes)
    )
    conn.commit()
    conn.close()


def remove_from_watchlist(text):
    conn = get_conn()
    conn.execute("DELETE FROM watchlist WHERE text = ?", (text,))
    conn.commit()
    conn.close()


def get_watchlist():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM watchlist ORDER BY added_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]
