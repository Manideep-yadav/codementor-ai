"""CodeMentor model: difficulty classifier + approach recommender + similar-problem retrieval."""
from __future__ import annotations

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from .data import Problem
from .features import DIFFICULTIES, PATTERNS, feature_matrix, rule_matrix, rule_scores, signals


class CodeMentor:
    """
    Two heads share one text representation (TF-IDF 1-2 grams + scaled handcrafted signals):

    * difficulty  - multinomial logistic regression -> P(easy/medium/hard)
    * approach    - one-vs-rest logistic regression, blended with an interpretable
                    keyword prior:  score = alpha * P_model + (1 - alpha) * P_rules
    Plus nearest-neighbour retrieval of similar known problems (cosine on TF-IDF).
    """

    def __init__(self, use_features: bool = True, alpha: float = 0.6, feat_weight: float = 0.5):
        self.use_features = use_features
        self.alpha = alpha
        self.feat_weight = feat_weight
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.scaler = StandardScaler()

    # ---- representation -------------------------------------------------
    def _design(self, texts: list[str], fit: bool = False):
        if fit:
            x_t = self.tfidf.fit_transform(texts)
            f = self.scaler.fit_transform(feature_matrix(texts))
        else:
            x_t = self.tfidf.transform(texts)
            f = self.scaler.transform(feature_matrix(texts))
        x = hstack([x_t, csr_matrix(f * self.feat_weight)]).tocsr() if self.use_features else x_t
        return x_t, x

    # ---- training -------------------------------------------------------
    def fit(self, problems: list[Problem]) -> "CodeMentor":
        texts = [p.text for p in problems]
        x_t, x = self._design(texts, fit=True)
        self.n_terms = x_t.shape[1]
        self.terms = np.array(self.tfidf.get_feature_names_out())
        self.diff_clf = LogisticRegression(C=5.0, max_iter=3000, class_weight="balanced")
        self.diff_clf.fit(x, [p.difficulty for p in problems])

        idx = [i for i, p in enumerate(problems) if p.patterns]
        self.mlb = MultiLabelBinarizer(classes=PATTERNS)
        y = self.mlb.fit_transform([problems[i].patterns for i in idx])
        self.pat_clf = OneVsRestClassifier(
            LogisticRegression(C=10.0, max_iter=3000, class_weight="balanced")).fit(x[idx], y)

        self._bank = [(p.title, p.difficulty, p.patterns) for p in problems]
        self._bank_x = x_t
        return self

    # ---- batch scoring (used by evaluation) -----------------------------
    def difficulty_proba(self, texts: list[str]) -> np.ndarray:
        _, x = self._design(texts)
        raw = self.diff_clf.predict_proba(x)
        order = [list(self.diff_clf.classes_).index(d) for d in DIFFICULTIES]
        return raw[:, order]

    def pattern_scores(self, texts: list[str], alpha: float | None = None) -> np.ndarray:
        a = self.alpha if alpha is None else alpha
        _, x = self._design(texts)
        model_p = self.pat_clf.predict_proba(x)
        return a * model_p + (1 - a) * rule_matrix(texts)

    # ---- single-problem inference with explanations ---------------------
    def predict(self, text: str, top_k: int = 3, n_similar: int = 3) -> dict:
        x_t, x = self._design([text])
        proba = self.difficulty_proba([text])[0]
        label = DIFFICULTIES[int(proba.argmax())]

        # Terms that pushed the score toward the predicted class.
        c = list(self.diff_clf.classes_).index(label)
        contrib = x_t.toarray().ravel() * self.diff_clf.coef_[c][: self.n_terms]
        why_terms = [str(self.terms[j]) for j in np.argsort(-contrib)[:5] if contrib[j] > 0]

        scores = self.pattern_scores([text])[0]
        rules = rule_scores(text)
        ranked = np.argsort(-scores)[:top_k]
        patterns = [{"pattern": PATTERNS[j], "score": round(float(scores[j]), 3),
                     "matched_cues": rules[PATTERNS[j]][1]} for j in ranked]

        sims = (x_t @ self._bank_x.T).toarray().ravel()
        similar = [{"title": self._bank[j][0], "difficulty": self._bank[j][1],
                    "patterns": list(self._bank[j][2]), "similarity": round(float(sims[j]), 3)}
                   for j in np.argsort(-sims)[:n_similar] if sims[j] > 0]

        return {"difficulty": {"label": label,
                               "probabilities": {d: round(float(p), 3) for d, p in zip(DIFFICULTIES, proba)},
                               "driving_terms": why_terms},
                "patterns": patterns, "similar_problems": similar,
                "signals": {k: round(v, 2) for k, v in signals(text).items()}}

    # ---- persistence ----------------------------------------------------
    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "CodeMentor":
        return joblib.load(path)
