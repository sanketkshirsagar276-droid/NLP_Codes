

import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# App Configuration
st.set_page_config(page_title="Restaurant Review Sentiment Analysis", layout="wide")
st.title("🍽 Sentiment Analysis - Restaurant Reviews (LightGBM)")
st.markdown("### Built by *Sanket* | NLP • ML • LightGBM • Streamlit")

# ---------------------------------------------------------
# Model file paths
MODEL_PATH = "sentiment_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

# ---------------------------------------------------------
# Load model & vectorizer if they exist
model = None
vectorizer = None
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    st.sidebar.success("✅ Pre-trained model loaded successfully!")
else:
    st.sidebar.warning("⚠ No saved model found. Please train one first.")

# ---------------------------------------------------------
# Dataset Section
st.sidebar.header("📂 Dataset Configuration")
use_default = st.sidebar.checkbox("Use Default Dataset", value=True)

if use_default:
    dataset_path = r"C:\Users\Sanket kshirsagar\Downloads\Restaurant_Reviews (1).tsv"
    dataset = pd.read_csv(dataset_path, delimiter='\t', quoting=3)
else:
    uploaded_file = st.sidebar.file_uploader("Upload TSV File", type=['tsv'])
    if uploaded_file is not None:
        dataset = pd.read_csv(uploaded_file, delimiter='\t', quoting=3)
    else:
        st.warning("Please upload a dataset or use the default one.")
        st.stop()

expand_data = st.sidebar.checkbox("Expand data (×20 for balance)", value=True)
if expand_data:
    dataset = pd.concat([dataset] * 36, ignore_index=True)

st.write("### Sample Data")
st.dataframe(dataset.head())

# ---------------------------------------------------------
# Preprocessing setup (no NLTK download)
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    st.error("❌ NLTK stopwords not found. Please run once:\n"
             ">>> import nltk\n>>> nltk.download('stopwords')")

try:
    lemmatizer = WordNetLemmatizer()
except LookupError:
    st.error("❌ WordNet not found. Please run once:\n"
             ">>> import nltk\n>>> nltk.download('wordnet')")

def clean_text(texts, remove_stopwords=True):
    corpus = []
    for text in texts:
        review = re.sub('[^a-zA-Z]', ' ', text)
        review = review.lower().split()
        if remove_stopwords:
            review = [lemmatizer.lemmatize(word) for word in review if word not in stop_words]
        else:
            review = [lemmatizer.lemmatize(word) for word in review]
        for j in range(len(review) - 1):
            if review[j] == 'not':
                review[j + 1] = 'not_' + review[j + 1]
        corpus.append(' '.join(review))
    return corpus

# ---------------------------------------------------------
# Train Model
if st.button("🚀 Train & Save LightGBM Model"):
    st.info("Cleaning text data...")
    corpus = clean_text(dataset['Review'])

    st.info("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=2500, ngram_range=(1, 2))
    X = vectorizer.fit_transform(corpus).toarray()
    y = dataset.iloc[:, 1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=10,
        num_leaves=31,
        random_state=42
    )

    with st.spinner("Training LightGBM model..."):
        model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    st.success("✅ Model trained successfully!")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy * 100:.2f}%")
    col2.metric("F1 Score", f"{f1 * 100:.2f}%")
    col3.metric("AUC", f"{auc:.3f}")

    st.subheader("📊 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    st.pyplot(fig)

    st.subheader("📋 Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Save model & vectorizer
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    st.success("💾 Model and vectorizer saved successfully!")

# ---------------------------------------------------------
# Single Review Prediction
st.divider()
st.subheader("💬 Try Single Review Prediction")

new_review = st.text_area("Enter a restaurant review:")
if st.button("Predict Sentiment"):
    if model is None or vectorizer is None:
        st.error("⚠ Please train or load a model first.")
    elif new_review.strip() == "":
        st.warning("Please enter a review text!")
    else:
        clean_rev = clean_text([new_review])
        X_new = vectorizer.transform(clean_rev).toarray()
        prediction = model.predict(X_new)[0]
        prob = model.predict_proba(X_new)[0][1]
        sentiment = "👍 Positive" if prediction == 1 else "👎 Negative"
        st.write(f"### Prediction: {sentiment}")
        st.progress(prob if prediction == 1 else 1 - prob)
        st.caption(f"Confidence: {prob*100:.2f}%" if prediction == 1 else f"Confidence: {(1-prob)*100:.2f}%")

# ---------------------------------------------------------
# Batch Prediction
st.divider()
st.subheader("📦 Batch Review Sentiment Prediction")

uploaded_reviews = st.file_uploader("Upload a CSV file with a 'Review' column:", type=['csv'])
if uploaded_reviews is not None:
    try:
        df_new = pd.read_csv(uploaded_reviews)

        if 'Review' not in df_new.columns:
            st.error("❌ The uploaded CSV must contain a column named 'Review'.")
        else:
            st.write("### Sample Uploaded Data")
            st.dataframe(df_new.head())

            if st.button("Predict Batch Sentiments"):
                if model is None or vectorizer is None:
                    st.error("⚠ Please train or load the model first.")
                else:
                    clean_reviews = clean_text(df_new['Review'])
                    X_new = vectorizer.transform(clean_reviews).toarray()
                    preds = model.predict(X_new)
                    probs = model.predict_proba(X_new)[:, 1]

                    df_new['Predicted_Sentiment'] = np.where(preds == 1, 'Positive', 'Negative')
                    df_new['Confidence'] = np.round(np.where(preds == 1, probs, 1 - probs) * 100, 2)

                    st.success("✅ Predictions generated successfully!")
                    st.dataframe(df_new.head())

                    csv = df_new.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💾 Download Predictions as CSV",
                        data=csv,
                        file_name='Predicted_Sentiments.csv',
                        mime='text/csv'
                    )
    except Exception as e:
        st.error(f"Error reading file: {e}")