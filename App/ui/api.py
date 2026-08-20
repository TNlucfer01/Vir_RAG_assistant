"""
ui/api.py — Backend API Client for the Streamlit UI

Functions:
  - upload_pdf(file)                     → POST /upload
  - ask_question_stream(...)             → POST /chat  (streaming generator)
  - get_suggestions(filename)            → POST /suggestions

Streaming is done by calling /chat normally but yielding tokens from the
response so Streamlit's st.write_stream() can render them as they arrive.
Since our backend isn't SSE-based, we fetch the full response and yield
the answer word-by-word with a tiny delay to simulate streaming.
(For true token streaming, the backend would need SSE — add later.)
"""

import os
import time
import json
import requests
import streamlit as st

try:
    BASE_URL = st.secrets.get("BACKEND_URL", "http://127.0.0.1:8000")
except Exception:
    BASE_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


# ── Upload ─────────────────────────────────────────────────────────────────────

def upload_pdf(uploaded_file):
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "application/octet-stream",
        )
    }
    response = requests.post(f"{BASE_URL}/upload", files=files, timeout=120)
    return response


# ── Chat ───────────────────────────────────────────────────────────────────────

def ask_question(
    question: str,
    filename: str,
    history: list,
    session_id: str = "",
) -> dict:
    """Calls the backend /chat endpoint and returns the parsed JSON dict."""
    payload = {
        "question": question,
        "filename": filename,
        "history": history,
        "session_id": session_id,
    }

    try:
        response = requests.post(
            f"{BASE_URL}/chat",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        return {"answer": "⚠️ The request timed out. Please try again."}
    except requests.ConnectionError:
        return {"answer": "⚠️ Could not connect to the backend. Make sure `uvicorn main:app` is running."}
    except Exception as e:
        return {"answer": f"⚠️ Error: {e}"}


# ── Suggestions ────────────────────────────────────────────────────────────────

def get_suggestions(filename: str):
    response = requests.post(
        f"{BASE_URL}/suggestions",
        json={"filename": filename},
        timeout=30,
    )
    return response