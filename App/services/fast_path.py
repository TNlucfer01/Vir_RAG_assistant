"""
services/fast_path.py — Lightweight Fast-Path Classifier

Handles crystal-clear queries without consuming the full agent loop.
Only two categories qualify for fast-path:
  1. sql_only   — bare 12-digit registration number
  2. map_only   — pure navigation query with no student/document component

Everything else returns None → falls through to the full agentic loop.
"""

import re

# ── Patterns ───────────────────────────────────────────────────────────────────

# Anna University 12-digit reg number (always → SQL)
_REG_NO_RE = re.compile(r"\b5\d{11}\b")

# Pure navigation phrases (→ map tools only)
_NAV_RE = re.compile(
    r"\b(navigate|how (do i|to) get (to|from)|directions? (to|from)|"
    r"find .{0,20}(room|lab|office|block|floor)|"
    r"path (from|to)|route (to|from)|"
    r"where is .{0,30}(room|lab|block|department|floor|building|toilet|canteen|library))\b",
    re.IGNORECASE,
)


def fast_path(question: str) -> str | None:
    """
    Returns:
        "sql_only"  — question definitely needs only SQL (reg number detected)
        "map_only"  — question is purely about campus navigation
        None        — ambiguous / complex → use full agent loop
    """
    q = question.strip()

    # Bare reg number → direct SQL lookup
    if _REG_NO_RE.search(q):
        return "sql_only"

    # Pure navigation (no student/document signals)
    if _NAV_RE.search(q):
        return "map_only"

    return None  # full agent loop
