# Cross-validated results (5-fold, n=2909, seed=42)

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
