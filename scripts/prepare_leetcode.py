"""Download a public LeetCode problem dump and convert it to data/leetcode.csv.

Source: github.com/neenza/leetcode-problems (merged_problems.json, ~2.9k problems).
Usage:
    python scripts/prepare_leetcode.py                 # download + convert
    python scripts/prepare_leetcode.py --input f.json  # convert a local copy
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/neenza/leetcode-problems/master/merged_problems.json"
DATA = Path(__file__).resolve().parent.parent / "data"

# The dump flattens superscripts: 10^4 -> "104", -10^9 -> "-109". Restore them.
_SUPER = re.compile(r"(?<![\d.^])10(1[0-8]|[4-9])(?!\d)")
_NOISE = re.compile(r"^\s*(Example\s*\d*:?|Constraints:?|Follow[- ]up:?)\s*$", re.I)


def fix_powers(s: str) -> str:
    return _SUPER.sub(r"10^\1", s)


def clean(q: dict) -> str:
    lines = [l for l in (q.get("description") or "").splitlines() if l.strip() and not _NOISE.match(l)]
    cons = [fix_powers(c) for c in (q.get("constraints") or [])]
    return " ".join(lines) + (" Constraints: " + "; ".join(cons) if cons else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="local merged_problems.json (skips the download)")
    ap.add_argument("--out", default=str(DATA / "leetcode.csv"))
    args = ap.parse_args()

    DATA.mkdir(exist_ok=True)
    raw = Path(args.input) if args.input else DATA / "leetcode_raw.json"
    if not args.input:
        print(f"downloading {URL} ...")
        urllib.request.urlretrieve(URL, raw)
    questions = json.loads(raw.read_text(encoding="utf-8"))["questions"]

    rows = 0
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["title", "description", "difficulty", "topics"])
        for q in questions:
            text = clean(q)
            if q.get("difficulty") and len(text) > 40:
                w.writerow([q["title"], text, q["difficulty"], ",".join(q.get("topics") or [])])
                rows += 1
    print(f"wrote {rows} problems -> {args.out}")


if __name__ == "__main__":
    main()
