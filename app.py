import os
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Clickbait Headline Detector",
    page_icon="📰",
    layout="wide"
)

MODEL_PATH = "models/best_model.pkl"
INFO_PATH = "models/model_info.json"
DATA_PATH = "data/processed/clickbait_dataset.csv"
RESULT_PATH = "reports/model_comparison.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

def predict_probability(model, text):
    # LinearSVC has decision_function; probability is converted with a sigmoid.
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([text])[0]
        return float(probs[1])
    score = float(model.decision_function([text])[0])
    return float(1 / (1 + __import__("math").exp(-score)))

st.title("📰 Clickbait Headline Detection System")
st.caption("Machine Learning + NLP + TF-IDF + Streamlit")

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Run prepare_data.py and train_models.py first.")
    st.stop()

model = load_model()
df = load_data()

with open(INFO_PATH) as f:
    info = json.load(f)

tab1, tab2, tab3 = st.tabs(["🔎 Detect Clickbait", "📊 Dashboard", "ℹ️ About"])

with tab1:
    st.subheader("Enter a News Headline")
    headline = st.text_area(
        "Headline",
        placeholder="Example: You Won't Believe What Happened Next!",
        height=120
    )

    if st.button("🔍 Check Headline", type="primary"):
        if not headline.strip():
            st.warning("Please enter a headline.")
        else:
            # The saved pipeline contains its own TF-IDF preprocessing.
            prediction = int(model.predict([headline])[0])
            probability = predict_probability(model, headline)

            if prediction == 1:
                st.error("⚠️ CLICKBAIT")
                st.metric("Clickbait Probability", f"{probability*100:.2f}%")
            else:
                st.success("✅ NOT CLICKBAIT")
                st.metric("Clickbait Probability", f"{probability*100:.2f}%")

            st.progress(min(max(probability, 0.0), 1.0))

            st.write("**Model used:**", info["best_model"])
            st.info(
                "This is a machine-learning classification based on patterns in "
                "the training dataset; it is not a factual or editorial judgment."
            )

with tab2:
    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Headlines", f"{len(df):,}")
    c2.metric("Clickbait", f"{int((df['Clickbait']==1).sum()):,}")
    c3.metric("Not Clickbait", f"{int((df['Clickbait']==0).sum()):,}")

    st.subheader("Model Comparison")
    if os.path.exists(RESULT_PATH):
        results = pd.read_csv(RESULT_PATH)
        st.dataframe(results, use_container_width=True)
        chart_df = results.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score"]]
        st.bar_chart(chart_df)

    st.subheader("Headline Examples")
    st.dataframe(
        df[["Headline", "Clickbait"]].sample(min(10, len(df)), random_state=42),
        use_container_width=True
    )

with tab3:
    st.subheader("About the Project")
    st.write("""
    This project detects whether a news headline is clickbait or not using
    Natural Language Processing and Machine Learning.

    Pipeline:
    Dataset → Cleaning → TF-IDF → ML Models → Evaluation → Best Model → Dashboard

    Models:
    - Logistic Regression
    - Multinomial Naive Bayes
    - Random Forest
    - Linear SVM

    Dataset:
    Kaggle Clickbait Dataset by Aman Anand.
    """)
