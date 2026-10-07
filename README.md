# 🧠 CodeMentor AI

An NLP-based **DSA problem difficulty estimator and algorithmic approach recommender**.
Paste a problem statement; get back:

1. **Difficulty**: P(easy / medium / hard) plus the terms that drove the call
2. **Approach**: top-3 patterns (DP, greedy, sliding window, two pointers, binary search, graph, tree, backtracking, stack, hashing, heap) with the matched cues
3. **Similar problems**: nearest known problems with their difficulty and patterns

```
$ python -m codementor predict "Given an array of integers and a target, count the contiguous
  subarrays whose sum equals the target. n <= 2 * 10^4, values may be negative."

Difficulty : MEDIUM   easy 12%  medium 81%  hard 7%
Approach   :
  1. hashing         0.44
  2. dp              0.24
  3. binary_search   0.06
Similar    :
  - Subarray Sum Count [medium; hashing]  sim=0.592
```

## Architecture

```
problem text
   |-- TF-IDF (1-2 grams, sublinear)  --+
   |-- handcrafted signals ------------+--> shared representation
   |     (constraint magnitude 10^k, optimisation/counting words,
   |      modulo, complexity hints, structure words, length)
   |
   |-- Difficulty head : multinomial logistic regression  -> P(easy/medium/hard)
   |-- Approach head   : one-vs-rest logistic regression
   |        score = a * P_model + (1 - a) * P_keyword_prior      (a = 0.6)
   '-- Retrieval       : cosine similarity over TF-IDF -> similar problems
```

Design choices: linear models keep it **explainable** (every prediction exposes the terms and cues behind it) and
suit small labelled data. Titles are deliberately **excluded** from model input because they leak technique names.
Constraint size (`10^5` vs `20`) is extracted as a feature since it is a strong complexity signal.

## Quickstart

```bash
pip install -r requirements.txt
python -m codementor train            # quick smoke test on the 69-problem seed set
python -m codementor evaluate         # 5-fold CV + baselines + ablations -> reports/
python -m codementor predict "..."    # CLI inference (--json for machine output)
streamlit run app.py                  # web demo
pytest                                # 9 tests
```

## Real dataset (recommended)

`scripts/prepare_leetcode.py` downloads a public dump of ~2.9k LeetCode problems
([neenza/leetcode-problems](https://github.com/neenza/leetcode-problems)) and writes `data/leetcode.csv`
(title, description + constraints, difficulty, topics). It also repairs the dump's flattened superscripts
(`10^4` appears as `104`), which would otherwise corrupt the constraint-size feature.
The CSV is git-ignored: the problem statements belong to LeetCode, so regenerate it locally instead of committing it.

```bash
python scripts/prepare_leetcode.py
python -m codementor train    --csv data/leetcode.csv
python -m codementor evaluate --csv data/leetcode.csv
```

Any other CSV with `title`, `description`, `difficulty`, `topics` columns also works (for example the Kaggle
"Leetcode Problem Dataset"). Platform tags are mapped onto the 11 patterns in `codementor/data.py::TAG_MAP`.

## Results (5-fold stratified CV on 2,909 real problems, seed 42)

## Difficulty

| Model | Accuracy | Macro-F1 | Easy<->Hard errors |
|---|---|---|---|
| full | 0.569 | 0.553 | 89 |
| tfidf_only | 0.557 | 0.537 | 110 |
| majority | 0.503 | 0.223 | 0 |

## Approach recommendation

| Scorer | hit@1 | hit@3 | recall@3 |
|---|---|---|---|
| rules_only | 0.41 | 0.651 | 0.511 |
| blend | 0.607 | 0.875 | 0.787 |
| model_only | 0.606 | 0.874 | 0.786 |
| freq_prior | 0.296 | 0.665 | 0.531 |
| tfidf_only_blend | 0.605 | 0.87 | 0.785 |

### Reading these numbers honestly

* **Difficulty:** 0.569 accuracy vs 0.503 for always predicting "medium", and macro-F1 0.553 vs 0.223.
  The handcrafted signals (mainly constraint size) cut severe easy<->hard mistakes from 110 to 89 versus TF-IDF alone.
  Predicting difficulty from text alone is intrinsically hard, since algorithmic depth is not visible in the wording.
* **Approach recommendation:** the correct technique is in the top 3 about 87.5% of the time (hit@3) vs 66.5% for a
  frequency-only baseline. Only problems with at least one mapped tag are scored (2,061 of 2,909).
* **The keyword rules do not generalise:** they scored 0.957 hit@1 on the 69-problem seed set but only 0.41 on real
  data, because I wrote the cues alongside that seed set (a circular evaluation). On real data the learned head does
  the work and the blend is about equal to the model alone, so the rules add little at this data scale.
* Tags are LeetCode's own, a noisy proxy for "the intended approach"; hit@k measures agreement with those tags.

## Roadmap (what would make this publishable-quality)

1. Hold out whole problem families (not random folds) to test generalisation.
2. Compare against sentence-embedding features (e.g. a MiniLM encoder) and a fine-tuned small transformer.
3. Ordinal loss for difficulty so easy-vs-hard mistakes cost more than adjacent-level ones.
4. Calibrate probabilities and add a "low confidence" abstain option.
5. Multi-label threshold tuning per pattern; add union-find, trie, bit manipulation, math.

## Layout

```
codementor/  data.py  features.py  model.py  evaluate.py  cli.py  seed_data.py
app.py       Streamlit demo
tests/       9 tests (features, schema, retrieval, persistence, CSV loader, CV sanity)
reports/     results.md  metrics.json  confusion_matrix.png
```
