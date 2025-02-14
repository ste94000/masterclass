import streamlit as st
import joblib
import os
import pandas as pd

model_path = "models/text_classification_model.pkl"
vectorizer_path = "models/tfidf_vectorizer.pkl"

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# ✅ Function to test the model
def predict_text(text):
    """ Predicts whether a given text is classified as label 0 or 1. """

    # Transform input text using TF-IDF vectorizer
    text_tfidf = vectorizer.transform([text])

    # Make prediction
    prediction = model.predict(text_tfidf)[0]

    # Map label to human-readable output
    if prediction == 0:
        return "😞 Fake"
    else: return "😊 Real"

# ✅ Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="💬", layout="centered")

st.title("💡 Fake News Detection")
st.markdown("### Enter an article below and check if it's fake now! 🚀")

# User input
user_input = st.text_area("📝 Enter your article here:", height=150, placeholder="Type or paste an article...")

# Predict sentiment
if st.button("🔍 Detect Fake News"):
    if user_input.strip():
        result = predict_text(user_input)
        if result != "Error":
            st.success(f"**Result :** {result}")
    else:
        st.warning("⚠️ Please enter an article before analyzing.")

# ✅ Extra UI enhancements
st.markdown("---")
st.markdown("### ✨ Why use this app?")
st.markdown("- 🔥 **Instant Fake News Detection** for your articles!")
st.markdown("- 🎨 **Beautiful & Minimal UI** for easy interaction.")
st.markdown("- 🚀 **Fast & Efficient** model built with TF-IDF & Logistic Regression.")
st.markdown("---")
st.markdown("💡 *Built with ❤️ using Streamlit and Machine Learning!* ✨")
