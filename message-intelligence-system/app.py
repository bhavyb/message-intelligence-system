"""Message Intelligence System - Streamlit app (Spam vs Legitimate).
Run:  streamlit run app.py
"""
import json, re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
st.set_page_config(page_title="Message Intelligence System", page_icon="📨", layout="wide")

SPAM_WORDS = ["free", "win", "won", "congratulations", "urgent", "claim", "limited offer", "limited time", "today only",
              "cashback", "earn money", "reward", "prize", "gift card", "exclusive", "bonus", "act fast", "get rich",
              "suspended", "brand new phone", "before it expires", "asap", "deal just for you"]
LEGIT_WORDS = ["please", "thanks", "thank you", "meeting", "report", "document", "proposal", "invoice", "review",
               "confirm", "let me know", "appreciate", "timeline", "schedule", "feedback", "notes", "discussion", "order"]


@st.cache_resource
def load():
    meta = json.load(open(BASE / "models" / "meta.json"))
    models = {"KNN": joblib.load(BASE / "models" / "knn.joblib"),
              "SVM (RBF)": joblib.load(BASE / "models" / "svm.joblib"),
              "Naive Bayes": joblib.load(BASE / "models" / "naive_bayes.joblib")}
    return meta, models


def keyword_scores(text):
    t = text.lower()
    return sum(w in t for w in SPAM_WORDS), sum(w in t for w in LEGIT_WORDS)


meta, models = load()
FEATURES = meta["features"]

st.title("📨 Message Intelligence System")
st.caption("Classifies a message as **Legitimate (0)** or **Spam (1)** using KNN, SVM and Naive Bayes.")

tab1, tab2, tab3 = st.tabs(["🔍 Predict a message", "📊 Model comparison", "📁 Batch prediction (CSV)"])

# ---------------------------------------------------------------- tab 1
with tab1:
    examples = {
        "Spam example": "URGENT: Claim your reward now instant cashback today only! http://bit.ly/5745",
        "Legit example": "Please find the attached invoice for the updated proposal. Thanks.",
        "Custom": "",
    }
    choice = st.selectbox("Pick an example or choose Custom", list(examples))
    text = st.text_area("Message text", value=examples[choice], height=90)

    auto_spam, auto_legit = keyword_scores(text)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("From the text (auto)")
        st.write(f"Length: **{len(text)}**  |  Words: **{len(text.split())}**")
        st.write(f"URLs: **{text.lower().count('http')}**  |  Digits: **{sum(ch.isdigit() for ch in text)}**")
        override = st.checkbox("Edit keyword scores manually")
        spam_kw = st.slider("Spam keyword score", 0, 5, auto_spam, disabled=not override)
        legit_kw = st.slider("Legit keyword score", 0, 5, auto_legit, disabled=not override)
        if not override:
            spam_kw, legit_kw = min(auto_spam, 5), min(auto_legit, 5)
    with c2:
        st.subheader("Sender behaviour")
        activity = st.slider("Sender activity score", 0.0, 100.0, 55.0)
        age = st.number_input("Sender account age (days)", 0, 3000, 300)
        sent24 = st.number_input("Messages sent in last 24h", 0, 200, 6)
    with c3:
        st.subheader("Time")
        hour = st.slider("Hour of day", 0, 23, 12)
        dow = st.selectbox("Day of week (0 = Monday)", list(range(7)), index=2)

    row = pd.DataFrame([{
        "message_length": len(text), "word_count": len(text.split()), "num_urls": text.lower().count("http"),
        "num_digits": sum(ch.isdigit() for ch in text), "spam_keyword_score": spam_kw, "legit_keyword_score": legit_kw,
        "sender_activity_score": activity, "sender_account_age_days": age, "messages_sent_last_24h": sent24,
        "hour_of_day": hour, "day_of_week": dow}])[FEATURES]

    model_name = st.radio("Model", list(models), horizontal=True)
    threshold = st.slider("Spam probability threshold (lower = catch more spam, higher = fewer false alarms)", 0.05, 0.95, 0.5, 0.05)

    if st.button("Classify", type="primary"):
        if not text.strip():
            st.warning("Please enter a message.")
        else:
            p = models[model_name].predict_proba(row)[0, 1]
            if p >= threshold:
                st.error(f"🚨 SPAM  —  P(Spam) = {p:.2%}")
            else:
                st.success(f"✅ LEGITIMATE  —  P(Spam) = {p:.2%}")
            st.progress(float(p))
            st.write("**All models:**")
            st.dataframe(pd.DataFrame({"Model": list(models),
                                       "P(Spam)": [m.predict_proba(row)[0, 1] for m in models.values()],
                                       "Prediction": ["Spam" if m.predict_proba(row)[0, 1] >= threshold else "Legit" for m in models.values()]}),
                         hide_index=True)
            with st.expander("Features sent to the model"):
                st.dataframe(row)

# ---------------------------------------------------------------- tab 2
with tab2:
    st.subheader("Test-set metrics")
    mt = pd.DataFrame(meta["metrics"]).T.drop(columns="Model", errors="ignore")
    st.dataframe(mt.style.format("{:.4f}"))
    st.bar_chart(mt)
    st.write(f"KNN best K = **{meta['best_k']}** | SVM params = {meta['svm_params']}")
    imgs = BASE / "images"
    for name in ["12_confusion_matrices.png", "14_roc.png", "15_pr_curves.png", "09_svm_margin_pca.png"]:
        if (imgs / name).exists():
            st.image(str(imgs / name), caption=name.replace("_", " ").replace(".png", ""))

# ---------------------------------------------------------------- tab 3
with tab3:
    st.write("Upload a CSV/XLSX containing these columns: `" + "`, `".join(FEATURES) + "`")
    f = st.file_uploader("File", type=["csv", "xlsx"])
    bm = st.selectbox("Model for batch", list(models), key="bm")
    if f is not None:
        data = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
        missing = [c for c in FEATURES if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            out = data.copy()
            out["P(Spam)"] = models[bm].predict_proba(data[FEATURES])[:, 1]
            out["Prediction"] = (out["P(Spam)"] >= 0.5).map({True: "Spam", False: "Legit"})
            st.write(out["Prediction"].value_counts())
            st.dataframe(out.head(200))
            st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv")
