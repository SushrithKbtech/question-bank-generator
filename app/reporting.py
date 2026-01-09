from __future__ import annotations
from collections import Counter


def compute_coverage_report(questions: list[dict]) -> dict:
    co_counts = Counter([q.get("co_mapping") for q in questions])
    bloom_counts = Counter([q.get("bloom_level") for q in questions])
    difficulty_counts = Counter([q.get("difficulty") for q in questions])

    return {
        "co_distribution": dict(co_counts),
        "bloom_distribution": dict(bloom_counts),
        "difficulty_distribution": dict(difficulty_counts),
        "total_questions": len(questions),
    }
