def get_top_k(question_type: str) -> int:
    """
    Decide how many chunks to retrieve based on
    the user's question type.
    """

    strategies = {
        "summary": 3,
        "comparison": 3,
        "list": 3,
        "explanation": 3,
        "general": 3,
        "definition": 3,
    }

    return strategies.get(question_type, 6)
