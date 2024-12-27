import streamlit as st
import pickle
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer

# Fungsi untuk preprocessing teks
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetical characters
    return text

# Load model dan vectorizer
try:
    with open('text_classifier.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except Exception as e:
    st.error(f"Failed to load model: {e}")

try:
    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except Exception as e:
    st.error(f"Failed to load vectorizer: {e}")

# Fungsi untuk prediksi sentimen
def predict_sentiment(text):
    """Memprediksi sentimen dari teks input."""
    try:
        # Preprocessing teks input
        cleaned_text = clean_text(text)
        transformed_text = vectorizer.transform([cleaned_text])  # Menggunakan vectorizer untuk transformasi
        prediction = model.predict(transformed_text)  # Model prediksi
        sentiment = "Positive" if prediction[0] == 1 else "Negative" if prediction[0] == 0 else "Neutral"
        return sentiment
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return "Error in prediction"

# Aplikasi Streamlit
st.title("Aplikasi Sentimen Ulasan Netflix")

# Input dari pengguna
input_text = st.text_area("Masukkan teks ulasan:", "Ketik ulasan di sini...")

# Tombol untuk prediksi
if st.button("Prediksi"):
    if input_text.strip():
        result = predict_sentiment(input_text)
        st.success(f"Hasil Prediksi: {result}")
    else:
        st.warning("Silakan masukkan teks untuk diprediksi.")

