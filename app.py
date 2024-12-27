import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Load pre-trained model (replace 'model.pkl' with your actual model file)
model_path = "model.pkl"  # Update with the correct path
model = joblib.load(model_path)

# Load dataset (replace 'dataset.csv' with your dataset file)
data_path = "dataset.csv"  # Update with the correct path
data = pd.read_csv(data_path)

# Streamlit app
def main():
    st.title("Sentiment Analysis: Netflix Reviews")

    st.sidebar.header("Navigation")
    options = ["Overview", "Predict Sentiment", "Explore Sentiments"]
    choice = st.sidebar.radio("Go to:", options)

    if choice == "Overview":
        st.header("Dataset Overview")
        st.write("### Sample Data")
        st.write(data.head())
        st.write("### Sentiment Distribution")
        sentiment_counts = data['sentiment'].value_counts()
        st.bar_chart(sentiment_counts)

    elif choice == "Predict Sentiment":
        st.header("Predict Sentiment of a Review")
        user_input = st.text_area("Enter a review:", "")

        if st.button("Predict"):
            if user_input:
                prediction = model.predict([user_input])[0]
                st.write(f"### Sentiment: {prediction}")
            else:
                st.write("Please enter a review to predict its sentiment.")

    elif choice == "Explore Sentiments":
        st.header("Explore Reviews by Sentiment")

        sentiment = st.selectbox("Select Sentiment", ["positive", "negative", "neutral"])
        filtered_data = data[data['sentiment'] == sentiment]

        st.write(f"### {sentiment.capitalize()} Reviews")
        st.write(filtered_data[['review']].head(20))

if __name__ == "__main__":
    main()
