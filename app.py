import streamlit as st
import pickle

# Load Model dan Vectorizer
with open('text_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

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
