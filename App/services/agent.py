"""
services/agent.py — Vir's Agentic Reasoning Loop

This is the core of the new architecture.

Instead of hard-coded routing (LOOKUP / COMPUTE / HYBRID via regex),
Vir now:
  1. Receives the question and conversation history.
  2. Calls Groq with ALL tool definitions exposed.
  3. The LLM decides which tools to call, in what order.
  4. Tool results are fed back as messages.
  5. The LLM iterates (up to MAX_ROUNDS) until it produces a final text answer.
  6. Returns the final answer string.

This gives the model full agency over retrieval strategy, with no manual routing.
"""

import json
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL
from services.agent_tools import AGENT_TOOL_DEFINITIONS, execute_agent_tool

_client = Groq(api_key=GROQ_API_KEY)

# Maximum reasoning rounds to prevent infinite loops
MAX_ROUNDS = 6

# ── System Prompt ──────────────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """You are Vir, the intelligent AI campus assistant for P.T. Lee Chengalvaraya Naicker College of Engineering and Technology (PT Lee CNCET).

You have access to the following tools:

1. **vector_search(query, top_k)** — Search uploaded documents (PDFs, regulations, policies, syllabus, transport schedules, prospectus, etc.) using semantic similarity.
   Use when: questions about academic rules, GPA/CGPA formulas, exam patterns, admission procedures, college facilities, regulations, course descriptions.

2. **sql_query(question)** — Query the campus SQLite database in natural language.
   Use when: questions about specific students (by name or reg no), marks, attendance, grades, arrears, GPA/CGPA values, faculty directory, batch statistics, department-wise data.

3. **find_path(source, destination)** — Get turn-by-turn navigation directions between two campus locations.
4. **list_rooms(query)** — Search for rooms, labs, offices, or facilities by name or category.
5. **get_room_info(room_id)** — Get details about a specific room.

## REASONING RULES

- **Think before calling tools.** Identify which tools are needed and in what order.
- **Call multiple tools when needed.** For example:
  - "Who has the highest marks and where is the exam office?" → sql_query + find_path
  - "Explain the GPA formula and tell me Aathi's current GPA" → vector_search + sql_query
- **If a tool returns no results,** try an alternate query (e.g., rephrase, use different keywords) before giving up.
- **Never fabricate data.** Only answer based on tool results. If tools return nothing relevant, say so honestly.
- **For aggregation questions** (average marks, top students, count by dept) → always use sql_query.
- **For concept/policy questions** (what is arrear, how is CGPA calculated) → always use vector_search.
- **For navigation** (how to get to room X, where is the library) → use find_path or list_rooms.
- **Keep responses concise and direct.** No meta-commentary like "I searched the database and found...".

## COLLEGE CONTEXT
- College: P.T. Lee Chengalvaraya Naicker College of Engineering and Technology
- Affiliated to: Anna University, Chennai
- Departments: CSE, IT, ECE, EEE, Mech, Civil, AI&DS
- Database contains: students from 2022–2026 batches, marks for IAT/model/university exams, attendance, faculty directory
"""


# ── Public API ─────────────────────────────────────────────────────────────────

def run_agent(question: str, history: list = None) -> dict:
    """
    Run Vir's full agentic reasoning loop.

    Args:
        question: The user's current question.
        history:  Prior conversation turns (list of {role, content} dicts).

    Returns:
        dict with keys:
            answer   — final answer string
            tools_used — list of tool names called
            rounds   — number of reasoning rounds used
    """
    if history is None:
        history = []

    # Build initial message list
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]

    # Include recent conversation history (last 6 turns)
    for msg in history[-6:]:
        if isinstance(msg, dict) and msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Append the current question
    messages.append({"role": "user", "content": question})

    tools_used = []
    rounds = 0

    print(f"\n{'='*60}")
    print(f"[Agent] Starting agentic loop for: {question[:100]}")
    print(f"[Agent] History turns: {len(history)}")

    for round_num in range(1, MAX_ROUNDS + 1):
        rounds = round_num
        print(f"\n[Agent] Round {round_num}/{MAX_ROUNDS} — {len(messages)} messages in context")

        try:
            response = _client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                tools=AGENT_TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=2048,
            )
        except Exception as e:
            print(f"[Agent] LLM call failed on round {round_num}: {e}")
            return {
                "answer": f"I encountered an error while processing your question: {e}",
                "tools_used": tools_used,
                "rounds": rounds,
            }

        choice = response.choices[0]
        message = choice.message

        # No tool calls → final text answer
        if not message.tool_calls:
            final_answer = message.content or ""
            print(f"[Agent] Final answer reached on round {round_num} ({len(final_answer)} chars)")
            print(f"[Agent] Tools used: {tools_used}")
            print(f"{'='*60}\n")
            return {
                "answer": final_answer,
                "tools_used": tools_used,
                "rounds": rounds,
            }

        # Log tool calls
        print(f"[Agent] LLM requested {len(message.tool_calls)} tool call(s):")
        for tc in message.tool_calls:
            print(f"  → {tc.function.name}({tc.function.arguments[:120]})")

        # Append assistant's tool-call message to the conversation
        messages.append({
            "role": "assistant",
            "content": message.content,  # may be None
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ],
        })

        # Execute each tool and append results
        for tc in message.tool_calls:
            tool_name = tc.function.name
            if tool_name not in tools_used:
                tools_used.append(tool_name)

            tool_result = execute_agent_tool(tool_name, tc.function.arguments)

            print(f"[Agent] {tool_name} result preview: {str(tool_result)[:200]}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(tool_result),
            })

    # Exhausted all rounds — do one final synthesis call without tools
    print(f"[Agent] Max rounds ({MAX_ROUNDS}) reached. Forcing final synthesis.")
    messages.append({
        "role": "user",
        "content": "Please synthesize a final answer based on the tool results above.",
    })

    try:
        final_response = _client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )
        final_answer = final_response.choices[0].message.content or \
            "I was unable to produce a complete answer. Please try rephrasing your question."
    except Exception as e:
        final_answer = f"I encountered an error in the final synthesis step: {e}"

    print(f"[Agent] Synthesized answer ({len(final_answer)} chars)")
    print(f"{'='*60}\n")
    return {
        "answer": final_answer,
        "tools_used": tools_used,
        "rounds": rounds,
    }
