import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================
# Load Models and Encoders
# =========================

@st.cache_resource
def load_model():
    with open('xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

@st.cache_resource
def load_sentence_transformer():
    # Load the pre-trained Sentence Transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Adjust if using a different model
    return model

# Load the models
model, label_encoder = load_model()
sentence_model = load_sentence_transformer()

# =========================
# Streamlit App Layout
# =========================

st.title("Medication Review Analyzer")

st.write("Enter your medication review below to analyze its category:")

user_input = st.text_area("Review Text", height=200)

if st.button("Analyze"):
    if user_input.strip():
        with st.spinner('Analyzing...'):
            # Preprocess the input text using Sentence Transformer
            input_embedding = sentence_model.encode([user_input])

            # Make prediction using the XGBoost model
            prediction = model.predict(input_embedding)
            predicted_category = label_encoder.inverse_transform(prediction)[0]

            # Get prediction probabilities
            prediction_proba = model.predict_proba(input_embedding)
            confidence = np.max(prediction_proba) * 100

            # Display the results
            st.success(f"**Predicted Category:** {predicted_category}")
            st.info(f"**Confidence:** {confidence:.2f}%")
    else:
        st.warning("Please enter a review text.")
