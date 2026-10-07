"""Command line interface: train / evaluate / predict."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import load_csv, load_seed
from .evaluate import run_cv, write_reports
from .model import CodeMentor

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "codementor.joblib"


def _data(args):
    return load_csv(args.csv) if args.csv else load_seed()


def _pretty(res: dict) -> str:
    d = res["difficulty"]
    out = [f"Difficulty : {d['label'].upper()}   " + "  ".join(f"{k} {v:.0%}" for k, v in d["probabilities"].items())]
    if d["driving_terms"]:
        out.append("  driven by: " + ", ".join(d["driving_terms"]))
    out.append("Approach   :")
    for i, p in enumerate(res["patterns"], 1):
        cues = f"  (cues: {', '.join(p['matched_cues'][:3])})" if p["matched_cues"] else ""
        out.append(f"  {i}. {p['pattern']:<15} {p['score']:.2f}{cues}")
    if res["similar_problems"]:
        out.append("Similar    :")
        for s in res["similar_problems"]:
            out.append(f"  - {s['title']} [{s['difficulty']}; {', '.join(s['patterns'])}]  sim={s['similarity']}")
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="codementor", description="DSA difficulty estimator + approach recommender")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "evaluate"):
        sp = sub.add_parser(name)
        sp.add_argument("--csv", help="CSV with title/description/difficulty/topics columns")
    ev = sub.choices["evaluate"]
    ev.add_argument("--folds", type=int, default=5)
    ev.add_argument("--seed", type=int, default=42)
    pr = sub.add_parser("predict")
    pr.add_argument("text", nargs="?", help="problem statement (or pipe via stdin)")
    pr.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "train":
        problems = _data(args)
        MODEL_PATH.parent.mkdir(exist_ok=True)
        CodeMentor().fit(problems).save(str(MODEL_PATH))
        print(f"trained on {len(problems)} problems -> {MODEL_PATH}")
    elif args.cmd == "evaluate":
        result = run_cv(_data(args), folds=args.folds, seed=args.seed)
        write_reports(result, ROOT / "reports")
        print((ROOT / "reports" / "results.md").read_text())
    else:
        text = args.text or sys.stdin.read()
        model = CodeMentor.load(str(MODEL_PATH)) if MODEL_PATH.exists() else CodeMentor().fit(load_seed())
        res = model.predict(text)
        print(json.dumps(res, indent=2) if args.json else _pretty(res))


if __name__ == "__main__":
    main()
