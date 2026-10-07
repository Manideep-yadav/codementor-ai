"""Stratified k-fold evaluation with baselines and ablations."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from .data import Problem
from .features import DIFFICULTIES, PATTERNS
from .model import CodeMentor

ALPHAS = {"rules_only": 0.0, "blend": 0.6, "model_only": 1.0}


def _hits(scores: np.ndarray, truth: np.ndarray, k: int) -> tuple[float, float]:
    """hit@k (any true pattern in top-k) and recall@k, averaged over rows."""
    top = np.argsort(-scores, axis=1)[:, :k]
    hit, rec = [], []
    for row, t in zip(top, truth):
        found = t[row].sum()
        hit.append(float(found > 0))
        rec.append(found / t.sum())
    return float(np.mean(hit)), float(np.mean(rec))


def run_cv(problems: list[Problem], folds: int = 5, seed: int = 42) -> dict:
    n = len(problems)
    y = np.array([p.difficulty for p in problems])
    truth = np.zeros((n, len(PATTERNS)))
    has_pat = np.array([bool(p.patterns) for p in problems])
    for i, p in enumerate(problems):
        for pat in p.patterns:
            truth[i, PATTERNS.index(pat)] = 1

    pred = {"full": np.empty(n, dtype=object), "tfidf_only": np.empty(n, dtype=object),
            "majority": np.empty(n, dtype=object)}
    pat = {name: np.zeros((n, len(PATTERNS))) for name in ALPHAS}
    pat["freq_prior"] = np.zeros((n, len(PATTERNS)))
    pat["tfidf_only_blend"] = np.zeros((n, len(PATTERNS)))

    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(np.zeros(n), y):
        train = [problems[i] for i in tr]
        test_texts = [problems[i].text for i in te]
        full = CodeMentor(use_features=True).fit(train)
        plain = CodeMentor(use_features=False).fit(train)
        pred["full"][te] = [DIFFICULTIES[j] for j in full.difficulty_proba(test_texts).argmax(1)]
        pred["tfidf_only"][te] = [DIFFICULTIES[j] for j in plain.difficulty_proba(test_texts).argmax(1)]
        pred["majority"][te] = Counter(y[tr]).most_common(1)[0][0]
        for name, a in ALPHAS.items():
            pat[name][te] = full.pattern_scores(test_texts, alpha=a)
        pat["tfidf_only_blend"][te] = plain.pattern_scores(test_texts)
        freq = truth[tr].sum(0)
        pat["freq_prior"][te] = freq

    diff = {}
    for name, yp in pred.items():
        diff[name] = {"accuracy": round(accuracy_score(y, yp), 3),
                      "macro_f1": round(f1_score(y, yp, average="macro", labels=DIFFICULTIES), 3),
                      "easy_hard_confusions": int(sum({a, b} == {"easy", "hard"} for a, b in zip(y, yp)))}
    cm = confusion_matrix(y, pred["full"], labels=DIFFICULTIES).tolist()

    patterns = {}
    for name, sc in pat.items():
        h1, r1 = _hits(sc[has_pat], truth[has_pat], 1)
        h3, r3 = _hits(sc[has_pat], truth[has_pat], 3)
        patterns[name] = {"hit@1": round(h1, 3), "hit@3": round(h3, 3), "recall@3": round(r3, 3)}

    # per-pattern AP-style view for the blended model: mean rank of true pattern
    per_pattern = {}
    for j, name in enumerate(PATTERNS):
        rows = np.where(truth[:, j] == 1)[0]
        ranks = [int((pat["blend"][r] > pat["blend"][r, j]).sum()) + 1 for r in rows]
        per_pattern[name] = {"n": len(rows), "median_rank": float(np.median(ranks))}

    return {"n_problems": n, "folds": folds, "seed": seed, "difficulty": diff,
            "confusion_matrix_full": {"labels": DIFFICULTIES, "matrix": cm},
            "patterns": patterns, "per_pattern": per_pattern}


def write_reports(result: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))

    d, p = result["difficulty"], result["patterns"]
    lines = [f"# Cross-validated results ({result['folds']}-fold, n={result['n_problems']}, seed={result['seed']})", "",
             "## Difficulty", "", "| Model | Accuracy | Macro-F1 | Easy<->Hard errors |", "|---|---|---|---|"]
    for k, v in d.items():
        lines.append(f"| {k} | {v['accuracy']} | {v['macro_f1']} | {v['easy_hard_confusions']} |")
    lines += ["", "## Approach recommendation", "", "| Scorer | hit@1 | hit@3 | recall@3 |", "|---|---|---|---|"]
    for k, v in p.items():
        lines.append(f"| {k} | {v['hit@1']} | {v['hit@3']} | {v['recall@3']} |")
    (out / "results.md").write_text("\n".join(lines) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cm = np.array(result["confusion_matrix_full"]["matrix"])
        fig, ax = plt.subplots(figsize=(4, 3.6))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3), DIFFICULTIES)
        ax.set_yticks(range(3), DIFFICULTIES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
        ax.set_title("Difficulty (CV, full model)")
        fig.tight_layout()
        fig.savefig(out / "confusion_matrix.png", dpi=160)
    except Exception:  # plotting is optional
        pass
