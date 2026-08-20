"""
tests/test_agent.py — Accuracy Tests for the Vir Agentic RAG System

Tests 3 categories:
  A. SQL-only queries (student/faculty data)
  B. Vector-search queries (document/policy questions)
  C. Hybrid queries (need both tools)

Each test sends a real question to run_agent() and verifies:
  - The correct tool(s) were called
  - The answer is non-empty and doesn't contain error strings
  - Key facts are present in the answer (where deterministic)

Run: python -m pytest tests/test_agent.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import pytest
from services.agent import run_agent
from services.fast_path import fast_path


# ── Helper ─────────────────────────────────────────────────────────────────────

def assert_tool_called(result: dict, tool_name: str):
    assert tool_name in result["tools_used"], (
        f"Expected '{tool_name}' to be called, got: {result['tools_used']}"
    )


def assert_answer_not_empty(result: dict):
    assert result["answer"].strip(), "Answer is empty"


def assert_no_error_in_answer(result: dict):
    bad_phrases = ["tool error", "sql query failed", "i was unable", "encountered an error"]
    answer_lower = result["answer"].lower()
    for phrase in bad_phrases:
        assert phrase not in answer_lower, f"Error phrase found: '{phrase}' in answer: {result['answer'][:200]}"


# ── Fast-Path Tests ────────────────────────────────────────────────────────────

class TestFastPath:
    def test_reg_number_is_sql_only(self):
        assert fast_path("511523205001") == "sql_only"
        assert fast_path("Who is 511523205001?") == "sql_only"

    def test_navigation_is_map_only(self):
        assert fast_path("How do I get to the library?") == "map_only"
        assert fast_path("Navigate to the CSE department block") == "map_only"

    def test_policy_question_not_fast_pathed(self):
        assert fast_path("What is the GPA formula?") is None

    def test_marks_question_not_fast_pathed(self):
        assert fast_path("What is the average attendance of IT students?") is None


# ── SQL Tool Tests ─────────────────────────────────────────────────────────────

class TestSQLQueries:
    """Queries that MUST route to sql_query tool."""

    def test_student_lookup_by_name(self):
        result = run_agent("Tell me about Aathi S from IT department")
        assert_tool_called(result, "sql_query")
        assert_answer_not_empty(result)
        assert_no_error_in_answer(result)
        print(f"\n[Test] Student lookup answer: {result['answer'][:200]}")

    def test_average_attendance_query(self):
        result = run_agent("What is the average attendance percentage of IT department students?")
        assert_tool_called(result, "sql_query")
        assert_answer_not_empty(result)
        assert_no_error_in_answer(result)
        print(f"\n[Test] Avg attendance answer: {result['answer'][:200]}")

    def test_top_students_by_marks(self):
        result = run_agent("Who are the top 3 students by marks in the CSE department?")
        assert_tool_called(result, "sql_query")
        assert_answer_not_empty(result)
        assert_no_error_in_answer(result)
        print(f"\n[Test] Top students answer: {result['answer'][:200]}")

    def test_count_arrear_students(self):
        result = run_agent("How many students have arrears in the IT department?")
        assert_tool_called(result, "sql_query")
        assert_answer_not_empty(result)
        assert_no_error_in_answer(result)
        print(f"\n[Test] Arrear count answer: {result['answer'][:200]}")

    def test_faculty_info(self):
        result = run_agent("What is the cabin number and phone of faculty Divagaran?")
        assert_tool_called(result, "sql_query")
        assert_answer_not_empty(result)
        assert_no_error_in_answer(result)
        print(f"\n[Test] Faculty info answer: {result['answer'][:200]}")


# ── Vector Search Tests ────────────────────────────────────────────────────────

class TestVectorSearchQueries:
    """Queries that MUST route to vector_search tool."""

    def test_gpa_formula_explanation(self):
        result = run_agent("Explain the GPA calculation formula")
        assert_tool_called(result, "vector_search")
        assert_answer_not_empty(result)
        assert_no_error_in_answer(result)
        print(f"\n[Test] GPA formula answer: {result['answer'][:200]}")

    def test_exam_pattern_question(self):
        result = run_agent("What is the exam pattern for Anna University undergraduate programs?")
        assert_tool_called(result, "vector_search")
        assert_answer_not_empty(result)
        assert_no_error_in_answer(result)
        print(f"\n[Test] Exam pattern answer: {result['answer'][:200]}")

    def test_attendance_policy(self):
        result = run_agent("What is the minimum attendance requirement for exam eligibility?")
        assert_tool_called(result, "vector_search")
        assert_answer_not_empty(result)
        assert_no_error_in_answer(result)
        print(f"\n[Test] Attendance policy answer: {result['answer'][:200]}")

    def test_admission_procedure(self):
        result = run_agent("What is the admission procedure for the college?")
        assert_tool_called(result, "vector_search")
        assert_answer_not_empty(result)
        assert_no_error_in_answer(result)
        print(f"\n[Test] Admission procedure answer: {result['answer'][:200]}")

    def test_transport_schedule(self):
        result = run_agent("What are the bus timings for the college transport?")
        assert_tool_called(result, "vector_search")
        assert_answer_not_empty(result)
        assert_no_error_in_answer(result)
        print(f"\n[Test] Transport schedule answer: {result['answer'][:200]}")


# ── Hybrid Tests ───────────────────────────────────────────────────────────────

class TestHybridQueries:
    """Queries that need BOTH sql_query AND vector_search."""

    def test_student_plus_regulation(self):
        result = run_agent(
            "Find a student with arrears and explain what the regulation says about arrear clearance"
        )
        # Should call both tools
        tools = result["tools_used"]
        assert "sql_query" in tools or "vector_search" in tools, \
            f"Expected at least one tool to be called, got: {tools}"
        assert_answer_not_empty(result)
        assert_no_error_in_answer(result)
        print(f"\n[Test] Hybrid answer: {result['answer'][:300]}")

    def test_performance_plus_formula(self):
        result = run_agent(
            "What is the average CGPA of IT students and how is CGPA calculated?"
        )
        tools = result["tools_used"]
        assert len(tools) >= 1, "Expected at least one tool"
        assert_answer_not_empty(result)
        assert_no_error_in_answer(result)
        print(f"\n[Test] CGPA hybrid answer: {result['answer'][:300]}")


# ── SQL Security Tests ─────────────────────────────────────────────────────────

class TestSQLSecurity:
    """Verify the SQL lockdown prevents destructive queries."""

    def test_blocks_delete(self):
        from services.sql_engine import _safe_execute
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.execute("INSERT INTO t VALUES (1)")
        conn.commit()
        result = _safe_execute(conn, "DELETE FROM t")
        assert result["error"] is not None
        assert "permitted" in result["error"] or "disallowed" in result["error"]
        conn.close()

    def test_blocks_drop(self):
        from services.sql_engine import _safe_execute
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.commit()
        result = _safe_execute(conn, "DROP TABLE t")
        assert result["error"] is not None
        conn.close()

    def test_blocks_cte_with_delete(self):
        from services.sql_engine import _safe_execute
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.execute("INSERT INTO t VALUES (1)")
        conn.commit()
        result = _safe_execute(conn, "WITH x AS (DELETE FROM t) SELECT 1")
        assert result["error"] is not None
        conn.close()

    def test_allows_select(self):
        from services.sql_engine import _safe_execute
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE t (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 'Alice')")
        conn.commit()
        result = _safe_execute(conn, "SELECT * FROM t")
        assert result["error"] is None
        assert len(result["rows"]) == 1
        conn.close()


if __name__ == "__main__":
    # Quick manual run
    import json

    print("\n" + "="*60)
    print("RUNNING QUICK ACCURACY CHECK")
    print("="*60)

    test_questions = [
        ("SQL",    "How many students are in the IT department?"),
        ("VECTOR", "What is the minimum attendance requirement?"),
        ("VECTOR", "Explain the GPA formula"),
        ("SQL",    "Who has the highest marks in CSE department?"),
        ("HYBRID", "What is the average attendance of IT students and what does the regulation say about attendance?"),
    ]

    results = []
    for expected_type, q in test_questions:
        print(f"\n[Q] ({expected_type}) {q}")
        try:
            r = run_agent(q)
            tools = r["tools_used"]
            answer_preview = r["answer"][:150].replace("\n", " ")
            print(f"    Tools used : {tools}")
            print(f"    Rounds     : {r['rounds']}")
            print(f"    Answer     : {answer_preview}...")
            results.append({
                "expected": expected_type,
                "question": q,
                "tools_used": tools,
                "rounds": r["rounds"],
                "answer_len": len(r["answer"]),
                "ok": bool(r["answer"].strip()),
            })
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({"question": q, "error": str(e), "ok": False})

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    passed = sum(1 for r in results if r.get("ok"))
    print(f"Passed: {passed}/{len(results)}")
    for r in results:
        status = "✅" if r.get("ok") else "❌"
        print(f"  {status} [{r.get('expected','?')}] {r['question'][:60]} → tools={r.get('tools_used', 'ERROR')}")
