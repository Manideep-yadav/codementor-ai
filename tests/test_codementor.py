import csv

import numpy as np
import pytest

from codementor import CodeMentor, load_csv, load_seed
from codementor.evaluate import run_cv
from codementor.features import DIFFICULTIES, PATTERNS, max_bound_exp, rule_scores


@pytest.fixture(scope="module")
def model():
    return CodeMentor().fit(load_seed())


def test_bound_extraction():
    assert max_bound_exp("1 <= n <= 10^5") == 5
    assert max_bound_exp("values up to 5 * 10^4 and 1e9") == 9
    assert max_bound_exp("n <= 100000") == pytest.approx(5)
    assert max_bound_exp("no numbers here") == 0
    assert max_bound_exp("string like 000 or 0000") == 0  # regression: log10(0) crash


def test_seed_is_well_formed():
    seed = load_seed()
    assert len(seed) >= 60
    assert {p.difficulty for p in seed} == set(DIFFICULTIES)
    assert all(pat in PATTERNS for p in seed for pat in p.patterns)


def test_rule_cues_fire():
    score, hits = rule_scores("Design a stack that supports push, pop and balanced parentheses")["stack"]
    assert score > 0.5 and hits


def test_predict_schema(model):
    res = model.predict("Count the number of ways to climb n stairs taking 1 or 2 steps. n <= 45.")
    assert res["difficulty"]["label"] in DIFFICULTIES
    assert abs(sum(res["difficulty"]["probabilities"].values()) - 1) < 0.02
    assert len(res["patterns"]) == 3
    assert res["patterns"][0]["score"] >= res["patterns"][1]["score"]


def test_obvious_patterns(model):
    top = lambda t: model.predict(t)["patterns"][0]["pattern"]
    assert top("Given the root of a binary tree, return the sum of all leaf values.") == "tree"
    assert top("Given a grid of letters, decide if a word can be traced through adjacent cells.") in {"backtracking", "graph"}
    assert top("Find the kth largest element in a stream of numbers using a heap.") == "heap"


def test_retrieval_finds_near_duplicate(model):
    seed = load_seed()[2]  # Balanced Brackets
    res = model.predict(seed.text)
    assert res["similar_problems"][0]["title"] == seed.title


def test_save_load_roundtrip(model, tmp_path):
    p = tmp_path / "m.joblib"
    model.save(str(p))
    again = CodeMentor.load(str(p))
    t = "Given a sorted array, find two numbers that sum to a target."
    assert again.predict(t)["difficulty"] == model.predict(t)["difficulty"]


def test_csv_loader_maps_tags(tmp_path):
    f = tmp_path / "d.csv"
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["title", "description", "difficulty", "topics"])
        w.writerow(["A", "Some statement n <= 10^5", "Medium", "Array,Dynamic Programming,Hash Table"])
        w.writerow(["B", "Other", "Weird", "Array"])  # bad difficulty -> skipped
    ps = load_csv(f)
    assert len(ps) == 1 and set(ps[0].patterns) == {"dp", "hashing"} and ps[0].difficulty == "medium"


def test_cv_beats_trivial_baselines():
    r = run_cv(load_seed(), folds=5, seed=42)
    assert r["difficulty"]["full"]["accuracy"] > r["difficulty"]["majority"]["accuracy"]
    assert r["patterns"]["blend"]["hit@3"] > r["patterns"]["freq_prior"]["hit@3"]
