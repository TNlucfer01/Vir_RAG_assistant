"""
services/session_store.py — Persistent Conversation Memory

Stores conversation history in SQLite, keyed by session_id.
Each session keeps the last 20 turns (pruned automatically).

Used by:
  - routes/chat.py — to load/save history per session
  - Streamlit passes session_id as a UUID generated at app startup
"""

import sqlite3
import json
import os
import time

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSION_DB = os.path.join(APP_DIR, "data", "sessions.db")
MAX_TURNS = 20       # max turns per session to store
PRUNE_AFTER_DAYS = 7 # auto-delete sessions older than this


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(SESSION_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id  TEXT NOT NULL,
            turn_index  INTEGER NOT NULL,
            role        TEXT NOT NULL,
            content     TEXT NOT NULL,
            created_at  REAL NOT NULL,
            PRIMARY KEY (session_id, turn_index)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_session_id ON sessions(session_id)
    """)
    conn.commit()
    return conn


def load_history(session_id: str) -> list[dict]:
    """
    Load the conversation history for a session.
    Returns a list of {role, content} dicts, ordered oldest first.
    """
    if not session_id:
        return []
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT role, content FROM sessions WHERE session_id = ? ORDER BY turn_index ASC",
            (session_id,),
        ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]
    finally:
        conn.close()


def save_turn(session_id: str, role: str, content: str):
    """
    Append one turn to the session history.
    Automatically prunes to the last MAX_TURNS turns.
    """
    if not session_id or not content:
        return
    conn = _get_conn()
    try:
        # Use a single atomic INSERT that calculates the next index inline
        conn.execute("""
            INSERT INTO sessions (session_id, turn_index, role, content, created_at)
            SELECT ?, COALESCE(MAX(turn_index), -1) + 1, ?, ?, ?
            FROM sessions
            WHERE session_id = ?
        """, (session_id, role, content, time.time(), session_id))
        conn.commit()

        # Prune to last MAX_TURNS
        conn.execute("""
            DELETE FROM sessions
            WHERE session_id = ?
              AND turn_index NOT IN (
                  SELECT turn_index FROM sessions
                  WHERE session_id = ?
                  ORDER BY turn_index DESC
                  LIMIT ?
              )
        """, (session_id, session_id, MAX_TURNS))
        conn.commit()
    finally:
        conn.close()


def save_exchange(session_id: str, question: str, answer: str):
    """Convenience: save user question + assistant answer as one exchange."""
    save_turn(session_id, "user", question)
    save_turn(session_id, "assistant", answer)


def clear_session(session_id: str):
    """Delete all history for a session."""
    if not session_id:
        return
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()
    finally:
        conn.close()


def prune_old_sessions():
    """Delete sessions older than PRUNE_AFTER_DAYS. Call periodically."""
    cutoff = time.time() - (PRUNE_AFTER_DAYS * 86400)
    conn = _get_conn()
    try:
        deleted = conn.execute(
            "DELETE FROM sessions WHERE created_at < ?", (cutoff,)
        ).rowcount
        conn.commit()
        if deleted:
            print(f"[SessionStore] Pruned {deleted} old session turns.")
    finally:
        conn.close()
