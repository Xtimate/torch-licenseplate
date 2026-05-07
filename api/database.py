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
       );

       CREATE TABLE IF NOT EXISTS review_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            crop BLOB NOT NULL,
            predicted_text TEXT,
            confidence REAL,
            source TEXT,
            status TEXT DEFAULT 'pending',
            labeled_text TEXT,
            added_at TEXT DEFAULT (datetime('now')),
            reviewed_at TEXT
       );

       CREATE TABLE IF NOT EXISTS model_versions (
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           filename TEXT NOT NULL,
           loss REAL,
           labeled_samples INTEGER,
           deployed_at TEXT DEFAULT (datetime('now')),
           is_active INTEGER DEFAULT 0
       );
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


def maybe_queue_for_review(
    predicted_text, confidence, source, crop_bytes, threshold=0.997
):
    if confidence >= threshold or crop_bytes is None:
        return
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO review_queue (crop, predicted_text, confidence, source)
        VALUES (?, ?, ?, ?)
    """,
        (crop_bytes, predicted_text, confidence, source),
    )
    conn.commit()
    conn.close()


def get_review_queue(limit=20, offset=0):
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, predicted_text, confidence, source, status, added_at FROM review_queue ORDER BY confidence ASC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_review_item_crop(item_id: int):
    conn = get_conn()
    row = conn.execute(
        "SELECT crop FROM review_queue WHERE id = ?",
        (item_id,),
    ).fetchone()
    conn.close()
    return row["crop"] if row else None


def label_review_item(item_id: int, labeled_text: str):
    conn = get_conn()
    conn.execute(
        """UPDATE review_queue SET status = 'labeled', labeled_text = ?, reviewed_at = datetime('now') WHERE id = ?""",
        (labeled_text.upper().strip(), item_id),
    )
    conn.commit()
    conn.close()


def reject_review_item(item_id: int):
    conn = get_conn()
    conn.execute(
        """UPDATE review_queue SET status = 'rejected', reviewed_at = datetime('now') WHERE id = ?""",
        (item_id,),
    )
    conn.commit()
    conn.close()


def get_labeled_items():
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, crop, labeled_text, confidence, source FROM review_queue WHERE status = 'labeled' ORDER BY added_at ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_queue_stats():
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) FROM review_queue").fetchone()[0]
    pending = conn.execute(
        "SELECT COUNT(*) FROM review_queue WHERE status = 'pending'"
    ).fetchone()[0]
    labeled = conn.execute(
        "SELECT COUNT(*) FROM review_queue WHERE status = 'labeled'"
    ).fetchone()[0]
    rejected = conn.execute(
        "SELECT COUNT(*) FROM review_queue WHERE status = 'rejected'"
    ).fetchone()[0]
    conn.close()
    return {
        "total": total,
        "pending": pending,
        "labeled": labeled,
        "rejected": rejected,
    }


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
        "SELECT country, COUNT(*) as count FROM plates GROUP BY country ORDER BY count DESC"
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


def get_analytics():
    conn = get_conn()
    avg_confidence = conn.execute("SELECT AVG(confidence) FROM plates").fetchone()[0]
    recent_24h = conn.execute(
        "SELECT COUNT(*) FROM plates WHERE timestamp >= datetime('now', '-24 hours')"
    ).fetchone()[0]
    watchlist_hits = conn.execute(
        "SELECT COUNT(*) FROM plates WHERE text IN (SELECT text FROM watchlist)"
    ).fetchone()[0]
    by_day = conn.execute(
        "SELECT strftime('%Y-%m-%d', timestamp) as day, COUNT(*) as count FROM plates GROUP BY day ORDER BY day DESC LIMIT 30"
    ).fetchall()
    confidence_buckets = conn.execute("""
        SELECT
            CASE
                WHEN confidence >= 0.95 THEN '0.95-1.00'
                WHEN confidence >= 0.90 THEN '0.90-0.95'
                WHEN confidence >= 0.80 THEN '0.80-0.90'
                WHEN confidence >= 0.70 THEN '0.70-0.80'
                ELSE '<0.70'
            END as bucket,
            COUNT(*) as count
        FROM plates
        GROUP BY bucket
        ORDER BY bucket DESC
    """).fetchall()
    top_plates = conn.execute(
        "SELECT text, country, COUNT(*) as count FROM plates GROUP BY text ORDER BY count DESC LIMIT 10"
    ).fetchall()
    by_hour = conn.execute(
        "SELECT strftime('%H', timestamp) as hour, COUNT(*) as count FROM plates GROUP BY hour ORDER BY hour"
    ).fetchall()
    conn.close()
    return {
        "recent_24h": recent_24h,
        "avg_confidence": round(avg_confidence, 4) if avg_confidence else None,
        "watchlist_hits": watchlist_hits,
        "by_day": [dict(r) for r in by_day],
        "confidence_buckets": [dict(r) for r in confidence_buckets],
        "top_plates": [dict(r) for r in top_plates],
        "by_hour": [dict(r) for r in by_hour],
    }


def register_model_version(filename: str, loss: float, labeled_samples: int):
    conn = get_conn()
    conn.execute("UPDATE model_versions SET is_active = 0")

    conn.execute(
        """
        INSERT INTO model_versions (filename, loss, labeled_samples, is_active)
        VALUES (?, ?, ?, 1)
        """,
        (filename, loss, labeled_samples),
    )
    conn.commit()
    conn.close()


def get_model_versions():
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM model_versions ORDER BY deployed_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_active_model_version():
    conn = get_conn()
    row = conn.execute("SELECT * FROM model_versions WHERE is_active = 1").fetchone()
    conn.close()
    return dict(row) if row else None


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
