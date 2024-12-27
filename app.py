import streamlit as st
import pandas as pd
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io

# Memuat dataset
df = pd.read_csv('com.netflix.mediaclient_reviews_min_100.csv')  # Ganti dengan path dataset Anda

# Fungsi untuk mendapatkan sentimen
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 5:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

# Menambahkan kolom sentimen
df['sentiment'] = df['content'].apply(get_sentiment)

# Mengencode sentimen ke bentuk numerik
label_encoder = LabelEncoder()
df['sentiment_encoded'] = label_encoder.fit_transform(df['sentiment'])

# Memisahkan ulasan positif dan negatif
positive_reviews = df[df['sentiment_encoded'] == 1]
negative_reviews = df[df['sentiment_encoded'] == 0]

# Fungsi untuk membuat word cloud
def create_wordcloud(text):
    return WordCloud(stopwords='english', background_color='white', width=800, height=400).generate(text)

# Aplikasi Streamlit
st.title("Aplikasi Sentimen Netflix")

# Input User
input_text = st.text_area("Masukkan teks ulasan:", "Ketik ulasan di sini...")

# Tombol Prediksi
if st.button("Prediksi"):
    if input_text.strip():
        # Prediksi sentimen
        sentiment = get_sentiment(input_text)
        st.success(f"Hasil Prediksi: {sentiment.capitalize()}")

        # Mengambil teks dari semua ulasan untuk word cloud
        positive_text = ' '.join(positive_reviews['content'])
        negative_text = ' '.join(negative_reviews['content'])

        # Membuat word cloud untuk positif dan negatif
        positive_wordcloud = create_wordcloud(positive_text)
        negative_wordcloud = create_wordcloud(negative_text)

        # Menampilkan word cloud
        st.subheader("Word Cloud Sentimen Positif")
        fig, ax = plt.subplots()
        ax.imshow(positive_wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

        st.subheader("Word Cloud Sentimen Negatif")
        fig, ax = plt.subplots()
        ax.imshow(negative_wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.warning("Silakan masukkan teks untuk diprediksi.")
