"""Streamlit demo:  streamlit run app.py"""
from pathlib import Path

import streamlit as st

from codementor import CodeMentor, load_seed

MODEL_PATH = Path(__file__).parent / "models" / "codementor.joblib"
EXAMPLE = ("Given a string s and an integer k, find the length of the longest substring "
           "that contains at most k distinct characters. 1 <= |s| <= 10^5.")


@st.cache_resource
def get_model() -> CodeMentor:
    return CodeMentor.load(str(MODEL_PATH)) if MODEL_PATH.exists() else CodeMentor().fit(load_seed())


st.set_page_config(page_title="CodeMentor AI", page_icon="🧠")
st.title("🧠 CodeMentor AI")
st.caption("Paste a DSA problem. Get a difficulty estimate, the likely approach, and similar problems.")

text = st.text_area("Problem statement", EXAMPLE, height=180)
if st.button("Analyse", type="primary") and text.strip():
    res = get_model().predict(text)
    d = res["difficulty"]
    st.subheader(f"Difficulty: {d['label'].upper()}")
    st.bar_chart(d["probabilities"])
    if d["driving_terms"]:
        st.write("Driven by:", ", ".join(f"`{t}`" for t in d["driving_terms"]))

    st.subheader("Recommended approach")
    for i, p in enumerate(res["patterns"], 1):
        st.progress(min(p["score"], 1.0), text=f"{i}. {p['pattern']}  ({p['score']:.2f})")
        if p["matched_cues"]:
            st.caption("cues: " + ", ".join(p["matched_cues"]))

    if res["similar_problems"]:
        st.subheader("Similar problems")
        for s in res["similar_problems"]:
            st.write(f"**{s['title']}** — {s['difficulty']} · {', '.join(s['patterns'])} · sim {s['similarity']}")
    with st.expander("Extracted signals"):
        st.json(res["signals"])
