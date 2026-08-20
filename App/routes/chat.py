"""
routes/chat.py — Vir Chat Endpoint (Agentic Architecture)

Flow:
  1. Fast-path check (lightweight classifier for obvious cases)
  2. Full agentic loop (model decides which tools to call)

The old hard-coded LOOKUP / COMPUTE / HYBRID 3-branch router has been replaced
by a single agent loop (services/agent.py) where the Groq LLM reasons about
which tools (vector_search, sql_query, map tools) to call.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from services.agent import run_agent
from services.agent_tools import AGENT_TOOL_DEFINITIONS, execute_agent_tool
from services.fast_path import fast_path
from services.sql_engine import run_sql
from services.followups import generate_followup_questions
from services.llm import generate_response
from services.map_tools import TOOL_DEFINITIONS as MAP_TOOL_DEFINITIONS, execute_tool as execute_map_tool
from services.session_store import load_history, save_exchange

import json
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL

router = APIRouter()
_client = Groq(api_key=GROQ_API_KEY)


# ── Fast-Path Handlers ─────────────────────────────────────────────────────────

def _handle_sql_only(question: str) -> str:
    """Direct SQL lookup for obvious queries (e.g., bare reg numbers)."""
    result = run_sql(question=question)
    if result.get("error"):
        return f"I tried to look that up but encountered a database error: {result['error']}"
    rows = result.get("rows", [])
    if not rows:
        return "I couldn't find any matching records in the database."
    rows_text = "\n".join(", ".join(f"{k}: {v}" for k, v in row.items()) for row in rows)
    format_prompt = (
        f"You are Vir, an AI campus assistant.\n\n"
        f"A database query was run for: {question}\n\n"
        f"Result:\n{rows_text}\n\n"
        f"Write a clear, concise natural-language answer. Do NOT mention SQL. Be direct."
    )
    return generate_response(format_prompt)


_MAP_ONLY_SYSTEM_PROMPT = """You are Vir, an intelligent campus assistant for
P.T. Lee Chengalvaraya Naicker College of Engineering and Technology.

Answer navigation questions using these tools:
- find_path(source, destination): Get shortest route + step-by-step directions.
- list_rooms(query): Search for rooms, labs, offices, or facilities.
- get_room_info(room_id): Get details about a specific room.

Always call the appropriate tool. Present directions clearly and concisely.
"""


def _handle_map_only(question: str, history: list) -> str:
    """Pure navigation queries using only map tools."""
    messages = [{"role": "system", "content": _MAP_ONLY_SYSTEM_PROMPT}]
    for msg in history[-4:]:
        if isinstance(msg, dict) and msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    for _ in range(5):
        response = _client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=MAP_TOOL_DEFINITIONS,
            tool_choice="auto",
            temperature=0.2,
        )
        msg = response.choices[0].message
        if not msg.tool_calls:
            return msg.content or ""
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in msg.tool_calls],
        })
        for tc in msg.tool_calls:
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": execute_map_tool(tc.function.name, tc.function.arguments)})

    return "I was unable to complete the navigation query. Please try rephrasing."


# ── Chat Request Model ─────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    filename: str = ""       # kept for API compatibility; agent searches all docs
    history: list = []
    session_id: str = ""     # optional persistent session ID for multi-turn memory


# ── Chat Endpoint ──────────────────────────────────────────────────────────────

@router.post("/chat")
async def chat(request: ChatRequest):
    question = request.question.strip()
    session_id = request.session_id.strip()

    # ── Load history: session store takes priority over inline history ─────────
    if session_id:
        history = load_history(session_id)
        print(f"[Chat] Session {session_id[:8]}… — loaded {len(history)} turns from store")
    else:
        history = request.history

    # ──────────────────────────────────────────────────────────────────────────
    # Fast-Path: handle obvious queries without the full agent loop
    # ──────────────────────────────────────────────────────────────────────────

    path = fast_path(question)

    if path == "sql_only":
        print(f"\n[Chat] Fast-path: sql_only")
        answer = _handle_sql_only(question)
        followups = generate_followup_questions(question=question, answer=answer)
        if session_id:
            save_exchange(session_id, question, answer)
        return {
            "question": question,
            "answer": answer,
            "followups": followups,
            "source": "fast_path_sql",
            "session_id": session_id,
            "debug": {"path": "sql_only"},
        }

    if path == "map_only":
        print(f"\n[Chat] Fast-path: map_only")
        answer = _handle_map_only(question, history)
        followups = generate_followup_questions(question=question, answer=answer)
        if session_id:
            save_exchange(session_id, question, answer)
        return {
            "question": question,
            "answer": answer,
            "followups": followups,
            "source": "fast_path_map",
            "session_id": session_id,
            "debug": {"path": "map_only"},
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Full Agentic Loop: model decides which tools to use
    # ──────────────────────────────────────────────────────────────────────────

    print(f"\n[Chat] Agentic loop for: {question[:80]}")

    result = run_agent(question=question, history=history)

    answer = result["answer"]
    tools_used = result["tools_used"]
    rounds = result["rounds"]

    if session_id:
        save_exchange(session_id, question, answer)

    followups = generate_followup_questions(question=question, answer=answer)

    print(f"[Chat] Tools used: {tools_used} | Rounds: {rounds}")

    return {
        "question": question,
        "answer": answer,
        "followups": followups,
        "source": "agent",
        "session_id": session_id,
        "debug": {
            "tools_used": tools_used,
            "rounds": rounds,
        },
    }
