import streamlit as st
import pickle
import logging

# Mengatur logging
logging.basicConfig(level=logging.ERROR)

# Menggunakan fungsi untuk memuat model dan vectorizer
def load_model(filename):
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        logging.error(f"Error loading {filename}: {e}")
        st.error(f"Failed to load the model: {e}")
        return None

# Memuat Model dan Vectorizer
model = load_model('text_classifier.pkl')
vectorizer = load_model('tfidf_vectorizer.pkl')

# Memastikan model dan vectorizer berhasil dimuat
if model is None or vectorizer is None:
    st.stop()  # Menghentikan aplikasi jika ada kesalahan

# Fungsi Prediksi
def predict_sentiment(text):
    """Memprediksi sentimen dari teks input."""
    transformed_text = vectorizer.transform([text])  # Preprocessing teks
    prediction = model.predict(transformed_text)  # Prediksi
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    return sentiment

# Aplikasi Streamlit
st.title("Aplikasi Sentimen Netflix")

# Input User
input_text = st.text_area("Masukkan teks ulasan:", "Ketik ulasan di sini...")

# Tombol Prediksi
if st.button("Prediksi"):
    if input_text.strip():
        result = predict_sentiment(input_text)
        st.success(f"Hasil Prediksi: {result}")
    else:
        st.warning("Silakan masukkan teks untuk diprediksi.")
