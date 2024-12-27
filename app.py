import streamlit as st
import pickle
import logging
import re

# Konfigurasi logging
logging.basicConfig(level=logging.ERROR)

# Fungsi untuk preprocessing teks
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub('@[^\s]+', '', text)  # Remove mentions
    text = re.sub('\d+', '', text)  # Remove digits
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'#\S+', '', text)  # Remove hashtags
    text = re.sub(r'\b[a-zA-Z]\b', '', text)  # Remove single characters
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Load model dan vectorizer
try:
    with open('text_classifier.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except Exception as e:
    logging.error(f"Error loading model: {e}")
    st.error(f"Failed to load the model: {e}")

try:
    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except Exception as e:
    logging.error(f"Error loading vectorizer: {e}")
    st.error(f"Failed to load the vectorizer: {e}")

# Fungsi untuk prediksi sentimen
def predict_sentiment(text):
    """Memprediksi sentimen dari teks input."""
    try:
        # Preprocessing teks input
        cleaned_text = clean_text(text)
        transformed_text = vectorizer.transform([cleaned_text])
        
        # Menampilkan bentuk data yang sudah diproses untuk debugging
        st.write(f"Transformed text shape: {transformed_text.shape}")  # Debugging
        prediction = model.predict(transformed_text)
        
        # Menampilkan prediksi raw untuk debugging
        st.write(f"Raw prediction: {prediction}")  # Debugging

        # Menentukan sentimen berdasarkan hasil prediksi
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        return sentiment
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return "Error in prediction"

# Aplikasi Streamlit
st.title("Aplikasi Sentimen Netflix")

# Input dari pengguna
input_text = st.text_area("Masukkan teks ulasan:", "Ketik ulasan di sini...")

# Tombol prediksi
if st.button("Prediksi"):
    if input_text.strip():
        result = predict_sentiment(input_text)
        st.success(f"Hasil Prediksi: {result}")
    else:
        st.warning("Silakan masukkan teks untuk diprediksi.")
