import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
import joblib
model = joblib.load("fake_news_model.joblib")

vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page configuration
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# --- Custom Styling ---
page_style = """
<style>
/* Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1f1c2c 0%, #928dab 100%);
    color: white;
}

/* Header */
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}

/* Headings */
h1 {
    text-align: center;
    color: #ffffff;
    text-shadow: 2px 2px 8px #000000;
    font-size: 2.5em;
}
h3 {
    text-align: center;
    color: #f0f0f0;
    margin-bottom: 20px;
}

/* Label Text */
label, .stTextArea label {
    color: #ffffff !important;
    font-weight: bold;
    font-size: 1.1em;
}

/* Text Area Fix */
.stTextArea textarea {
    background-color: rgba(255, 255, 255, 0.9); /* Light white box */
    color: #000000; /* Black text */
    border-radius: 10px;
    border: 1px solid #bbb;
    font-size: 1.05em;
}
.stTextArea textarea::placeholder {
    color: #555555; /* Gray placeholder */
}

/* Button styling */
.stButton button {
    background: linear-gradient(90deg, #ff512f, #dd2476);
    color: white;
    font-weight: bold;
    border-radius: 10px;
    padding: 10px 25px;
    border: none;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
}
.stButton button:hover {
    background: linear-gradient(90deg, #ff7e5f, #feb47b);
    color: black;
}

/* Result box */
.result-box {
    background-color: rgba(255,255,255,0.15);
    border-radius: 15px;
    padding: 15px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
    margin-top: 20px;
    backdrop-filter: blur(5px);
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# --- App Content ---
st.markdown("<h1>📰 Fake News Detection System</h1>", unsafe_allow_html=True)
st.markdown("<h3>Detect whether a news article is REAL or FAKE using Machine Learning 🧠</h3>", unsafe_allow_html=True)

# Input text
news_text = st.text_area("Enter News Content Below 👇", height=150, placeholder="Type or paste the news article here...")

# Predict button
if st.button("Check News Authenticity"):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        transformed_input = vectorizer.transform([news_text])
        prediction = model.predict(transformed_input)[0]
        confidence = model.predict_proba(transformed_input).max() * 100

        if prediction == "FAKE":
            st.markdown(
                f"<div class='result-box' style='color:#ff6b6b;'>🚨 This News Seems **FAKE** ({confidence:.2f}% confidence)</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='result-box' style='color:#4caf50;'>✅ This News Seems **REAL** ({confidence:.2f}% confidence)</div>",
                unsafe_allow_html=True,
            )

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#f0f0f0;'>Made with ❤️ by <b>Tejaswini</b> | Machine Learning Project</p>",
    unsafe_allow_html=True,
)
