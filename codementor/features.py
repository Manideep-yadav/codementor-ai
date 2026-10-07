"""Hand-crafted signals and keyword cues for DSA problem statements."""
from __future__ import annotations

import math
import re

import numpy as np

PATTERNS = ["dp", "greedy", "sliding_window", "two_pointers", "binary_search",
            "graph", "tree", "backtracking", "stack", "hashing", "heap"]
DIFFICULTIES = ["easy", "medium", "hard"]

# Plain-phrase cues (substring match on lowercased text). They act as an
# interpretable prior that is blended with the learned classifier.
PATTERN_CUES = {
    "dp": ["number of ways", "distinct ways", "fewest", "longest increasing", "subsequence",
           "minimum number of", "maximum amount", "partitioned into two", "convert one into",
           "maximum coins", "best order", "well-formed", "unique paths", "cuts"],
    "greedy": ["as many as", "largest number of", "jump", "starting station", "cooldown",
               "minimum number of intervals", "largest profit", "at most two people", "circuit"],
    "sliding_window": ["contiguous subarray of length", "window", "substring", "block of consecutive",
                       "every window", "longest substring", "shortest substring"],
    "two_pointers": ["sorted in ascending", "two numbers", "triplets", "merge them", "palindrome",
                     "reads the same", "two lines", "in ascending order"],
    "binary_search": ["o(log", "rotated", "minimum integer", "sorted array of distinct", "median of the combined",
                      "would be inserted", "unknown pivot", "strictly increases and then"],
    "graph": ["graph", "edge list", "grid", "adjacent", "connected", "prerequisite", "cycle",
              "shortest sequence", "flights", "cities", "alphabet", "connect all points", "path exists",
              "path between"],
    "tree": ["binary tree", "binary search tree", "leaf", "ancestor", "root of", "serialize", "nodes on the longest path"],
    "backtracking": ["all possible", "every binary string", "orderings", "unique combinations", "queens",
                     "all distinct solutions", "exactly once", "reused any number", "traced through"],
    "stack": ["brackets", "parentheses", "nested", "histogram", "warmer", "push, pop", "elevation map", "stack"],
    "hashing": ["appears", "anagram", "repeat", "distinct", "more than n/2", "frequent", "sum equals k",
                "duplicates", "rearranged"],
    "heap": ["kth largest", "k most frequent", "k sorted", "median", "stream", "heaviest",
             "top k", "k closest", "smallest k"],
}

_OPT = ["minimum", "maximum", "fewest", "shortest", "longest", "smallest", "largest", "cheapest", "most", "best"]
_COUNT = ["number of ways", "count", "distinct ways", "how many", "all possible", "return all", "generate every"]
_STRUCT = ["graph", "tree", "grid", "matrix", "linked list", "heap", "stack", "node", "edge", "board"]
_SUBSEQ = ["subsequence", "substring", "subarray", "partition", "split"]
_DESIGN = ["design", "implement", "serialize", "deserialize", "support"]
_MOD = re.compile(r"modulo|10\^9\s*\+\s*7")
_LOG = re.compile(r"o\(\s*log|o\(\s*n\s*\)|constant time")
_BOUND = re.compile(r"10\s*\^\s*(\d+)|10\s*\*\*\s*(\d+)|\b1e(\d+)\b|\b(\d{1,3}(?:,\d{3})+|\d{3,})\b", re.I)

FEATURE_NAMES = ["log_chars", "n_sentences", "max_bound_exp", "n_numbers", "optimisation_words",
                 "counting_words", "has_modulo", "has_complexity_hint", "structure_words",
                 "subsequence_words", "design_words"]


def max_bound_exp(text: str) -> float:
    """log10 of the largest constraint bound mentioned (e.g. 10^5 -> 5.0)."""
    best = 0.0
    for m in _BOUND.finditer(text):
        exp = m.group(1) or m.group(2) or m.group(3)
        if exp:
            best = max(best, float(exp))
        else:
            value = int(m.group(4).replace(",", ""))
            if value > 0:  # tokens such as "000" appear in sample strings
                best = max(best, math.log10(value))
    return best


def _count(words: list[str], t: str) -> int:
    return sum(t.count(w) for w in words)


def feature_row(text: str) -> list[float]:
    t = text.lower()
    return [math.log1p(len(t)), t.count(". ") + 1, max_bound_exp(text), len(re.findall(r"\d+", t)),
            _count(_OPT, t), _count(_COUNT, t), float(bool(_MOD.search(t))), float(bool(_LOG.search(t))),
            _count(_STRUCT, t), _count(_SUBSEQ, t), _count(_DESIGN, t)]


def feature_matrix(texts: list[str]) -> np.ndarray:
    return np.array([feature_row(t) for t in texts], dtype=float)


def signals(text: str) -> dict:
    return dict(zip(FEATURE_NAMES, feature_row(text)))


def rule_scores(text: str) -> dict[str, tuple[float, list[str]]]:
    """Per-pattern (score in [0,1), matched cues). Each extra cue halves the remaining doubt."""
    t = text.lower()
    out = {}
    for p in PATTERNS:
        hits = [c for c in PATTERN_CUES[p] if c in t]
        out[p] = (1 - 0.5 ** len(hits), hits)
    return out


def rule_matrix(texts: list[str]) -> np.ndarray:
    return np.array([[rule_scores(t)[p][0] for p in PATTERNS] for t in texts])
