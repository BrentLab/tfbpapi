import os
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE

CACHE_DIR = Path(os.getenv("HF_CACHE_DIR", HF_HUB_CACHE))


def get_hf_token() -> str | None:
    """Get HuggingFace token from environment variable."""
    return os.getenv("HF_TOKEN")


SQL_FILTER_KEYWORDS = sql_keywords = {
    "AND",
    "OR",
    "NOT",
    "IN",
    "IS",
    "NULL",
    "TRUE",
    "FALSE",
    "LIKE",
    "BETWEEN",
    "EXISTS",
    "ALL",
    "ANY",
    "SOME",
    "CASE",
    "WHEN",
    "THEN",
    "ELSE",
    "END",
    "CAST",
    "AS",
    "SELECT",
    "FROM",
    "WHERE",
    "GROUP",
    "ORDER",
    "BY",
    "HAVING",
    "LIMIT",
    "OFFSET",
    "DISTINCT",
    "COUNT",
    "SUM",
    "AVG",
    "MIN",
    "MAX",
    "UPPER",
    "LOWER",
    "SUBSTR",
    "LENGTH",
}
