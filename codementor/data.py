"""Dataset loading: bundled seed set or any CSV (e.g. a LeetCode Kaggle dump)."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from .features import DIFFICULTIES
from .seed_data import SEED


@dataclass(frozen=True)
class Problem:
    title: str
    text: str
    difficulty: str
    patterns: tuple[str, ...]


# Maps common platform tags onto CodeMentor's pattern vocabulary.
TAG_MAP = {
    "dynamic programming": "dp", "dp": "dp", "greedy": "greedy",
    "sliding window": "sliding_window", "two pointers": "two_pointers",
    "binary search": "binary_search", "graph": "graph", "breadth-first search": "graph",
    "depth-first search": "graph", "union find": "graph", "topological sort": "graph",
    "shortest path": "graph", "tree": "tree", "binary tree": "tree", "binary search tree": "tree",
    "backtracking": "backtracking", "stack": "stack", "monotonic stack": "stack",
    "hash table": "hashing", "hashing": "hashing", "heap (priority queue)": "heap",
    "heap": "heap", "priority queue": "heap",
}
_TEXT_COLS = ("description", "text", "content", "problem", "statement", "question")
_TAG_COLS = ("topics", "tags", "related_topics", "topic")


def load_seed() -> list[Problem]:
    return [Problem(t, x, d, tuple(p)) for t, x, d, p in SEED]


def _pick(row: dict, names) -> str:
    for n in names:
        for k in row:
            if k and k.strip().lower() == n and row[k]:
                return row[k]
    return ""


def load_csv(path: str | Path) -> list[Problem]:
    """Load problems from a CSV with title/description/difficulty/topics columns."""
    problems = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            diff = _pick(row, ("difficulty",)).strip().lower()
            text = _pick(row, _TEXT_COLS).strip()
            if diff not in DIFFICULTIES or not text:
                continue
            tags = re.split(r"[,|;]", _pick(row, _TAG_COLS))
            pats = tuple(sorted({TAG_MAP[t.strip().lower()] for t in tags if t.strip().lower() in TAG_MAP}))
            problems.append(Problem(_pick(row, ("title", "name")) or "untitled", text, diff, pats))
    return problems
