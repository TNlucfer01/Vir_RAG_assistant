"""
services/agent_tools.py — Tool Definitions + Executor for the Vir Agent

Exposes three categories of tools to the Groq LLM:
  1. vector_search  — semantic search in Qdrant (documents, PDFs, regulations)
  2. sql_query      — natural-language SQL against the campus SQLite database
  3. Map tools      — find_path, list_rooms, get_room_info (campus navigation)

The LLM receives AGENT_TOOL_DEFINITIONS and calls execute_agent_tool() for
each tool call it makes.
"""

import json
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception, before_sleep_log
from services.retriever import retrieve_context
from services.vectordb import search_embeddings, client as qdrant_client, COLLECTION_NAME
from services.embeddings import generate_query_embedding
from services.sql_engine import run_sql
from services.map_tools import execute_tool as execute_map_tool, TOOL_DEFINITIONS as MAP_TOOL_DEFINITIONS

logger = logging.getLogger(__name__)


def _network_retryable(exc: BaseException) -> bool:
    import requests
    return isinstance(exc, (requests.ConnectionError, requests.Timeout, TimeoutError, ConnectionError))




# ── Tool Definitions (sent to Groq) ────────────────────────────────────────────

_VECTOR_SEARCH_DEF = {
    "type": "function",
    "function": {
        "name": "vector_search",
        "description": (
            "Search the document knowledge base using semantic similarity. "
            "Use this for questions about regulations, policies, syllabus, exam patterns, "
            "admission procedures, GPA/CGPA formulas, college prospectus, transport schedule, "
            "academic rules, and any uploaded PDF content. "
            "Do NOT use this for specific student records, marks, attendance, or faculty contact info."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search query. Be specific — rephrase as a descriptive phrase "
                        "to improve semantic match (e.g. 'GPA calculation formula for arrear students')."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of document chunks to retrieve (default 8, max 15).",
                    "default": 8,
                },
            },
            "required": ["query"],
        },
    },
}

_SQL_QUERY_DEF = {
    "type": "function",
    "function": {
        "name": "sql_query",
        "description": (
            "Query the campus SQLite database with a natural language question. "
            "Use this for: student records (name, reg no, dept, batch, blood group, contact), "
            "marks and grades (IAT, model exams, university results), "
            "attendance and eligibility, GPA/CGPA values, arrears, "
            "faculty directory (name, cabin, phone, designation), "
            "batch statistics and aggregations. "
            "Do NOT use this for policy/regulation/concept questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Natural language question about student or faculty data.",
                },
            },
            "required": ["question"],
        },
    },
}

# Combine: vector_search + sql_query + all map tools
AGENT_TOOL_DEFINITIONS = [
    _VECTOR_SEARCH_DEF,
    _SQL_QUERY_DEF,
    *MAP_TOOL_DEFINITIONS,
]


# ── Tool Executor ──────────────────────────────────────────────────────────────

def execute_agent_tool(tool_name: str, arguments_json: str) -> str:
    """
    Dispatch a tool call from the LLM.

    Args:
        tool_name:       The name of the tool to execute.
        arguments_json:  JSON string of tool arguments.

    Returns:
        A string result to be sent back to the LLM as a tool message.
    """
    try:
        args = json.loads(arguments_json) if isinstance(arguments_json, str) else arguments_json
    except json.JSONDecodeError as e:
        return f"[Tool Error] Invalid JSON arguments: {e}"

    print(f"\n[AgentTools] Executing tool: {tool_name}")
    print(f"[AgentTools] Arguments: {args}")

    # ── vector_search ──────────────────────────────────────────────────────────
    if tool_name == "vector_search":
        query = args.get("query", "")
        top_k = min(int(args.get("top_k", 8)), 15)
        if not query:
            return "[Tool Error] vector_search requires a 'query' argument."
        try:
            # Get query embedding
            query_embedding = generate_query_embedding(query)

            # Search Qdrant directly so we can access filename + page metadata
            from services.vectordb import search_embeddings
            results = search_embeddings(
                query_embedding=query_embedding,
                filename=None,
                top_k=top_k,
            )

            documents = results.get("documents", [[]])[0]
            pages = results.get("pages", [])
            filenames = results.get("filenames", [])

            if not documents:
                return "No relevant documents found for this query."

            # De-duplicate chunks
            from services.context_filter import remove_duplicate_chunks
            unique_docs = remove_duplicate_chunks(documents)

            # Build context with inline source markers
            context_parts = []
            sources_seen = {}   # filename → set of pages
            for doc, page, fname in zip(documents, pages, filenames):
                if doc not in unique_docs:
                    continue
                tag = f"[{fname}, p.{page}]" if page else f"[{fname}]"
                context_parts.append(f"{tag}\n{doc}")
                if fname not in sources_seen:
                    sources_seen[fname] = set()
                if page:
                    sources_seen[fname].add(page)

            # Assemble context (cap at 8000 chars)
            context = "\n\n".join(context_parts)[:8000]

            # Append a clean SOURCES block for the model to cite
            sources_block = "\n\nSOURCES:\n" + "\n".join(
                f"- {fname} (pages: {', '.join(str(p) for p in sorted(pages_set))})"
                if pages_set else f"- {fname}"
                for fname, pages_set in sources_seen.items()
            )

            full_result = context + sources_block
            print(f"[AgentTools] vector_search returned {len(full_result)} chars, sources: {list(sources_seen.keys())}")
            return full_result

        except Exception as e:
            return f"[Tool Error] vector_search failed: {e}"

    # ── sql_query ──────────────────────────────────────────────────────────────
    elif tool_name == "sql_query":
        question = args.get("question", "")
        if not question:
            return "[Tool Error] sql_query requires a 'question' argument."
        try:
            result = run_sql(question=question)
            if result.get("error"):
                return f"SQL query failed: {result['error']}"
            rows = result.get("rows", [])
            if not rows:
                return "SQL query returned no results."
            # Format rows as readable text
            lines = []
            for row in rows:
                lines.append(", ".join(f"{k}: {v}" for k, v in row.items()))
            sql_text = result.get("sql", "")
            output = f"SQL: {sql_text}\n\nResults ({len(rows)} rows):\n" + "\n".join(lines)
            print(f"[AgentTools] sql_query returned {len(rows)} rows")
            return output
        except Exception as e:
            return f"[Tool Error] sql_query failed: {e}"

    # ── Map tools (find_path, list_rooms, get_room_info) ──────────────────────
    elif tool_name in ("find_path", "list_rooms", "get_room_info"):
        return execute_map_tool(tool_name, arguments_json)

    else:
        return f"[Tool Error] Unknown tool: {tool_name}"
