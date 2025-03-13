import streamlit as st
import pandas as pd
import joblib
import requests
import os
from dotenv import load_dotenv  # Load environment variables

# Load trained classifier and vectorizer
vectorizer = joblib.load("tfidf_vectorizer.joblib")
model = joblib.load("news_classifier_model.joblib")

# Load environment variables from .env
load_dotenv()

# Get API key from .env
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Streamlit UI
st.title("📰 Fake News Classifier")
st.write("🔍 Enter a topic to fetch **real-time news** and classify it as True or Fake.")

# Display error if API key is missing
if not NEWS_API_KEY:
    st.error("⚠️ API key is missing. Please add it to your `.env` file.")

# News API URL
NEWS_API_URL = "https://newsapi.org/v2/everything"

# Function to fetch live news
def fetch_news(query):
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": NEWS_API_KEY
    }
    response = requests.get(NEWS_API_URL, params=params)
    if response.status_code == 200:
        return response.json().get("articles", [])
    else:
        st.error("⚠️ Failed to fetch news. Try again later.")
        return []

# Function to classify the news as True or Fake
def classify_article(content):
    if content:
        # Convert the article text into a TF-IDF vector
        transformed_content = vectorizer.transform([content])
        
        # Predict (0 = Fake, 1 = True)
        prediction = model.predict(transformed_content)
        
        return prediction[0]  # Return the prediction (0 or 1)
    return None  # Return None if there's no content

# User input for search query
search_query = st.text_input("🔎 Enter a keyword:", "")

# Toggle for True or Fake news
news_type = st.radio("📢 Show:", ("True News", "Fake News"))

# Search button
button = st.button("Search Articles")

if button and search_query:
    st.info("🔄 Fetching news articles...")

    # Fetch live news
    articles = fetch_news(search_query)

    if not articles:
        st.warning("⚠️ No articles found for your search.")
    else:
        # Filter articles based on classification
        matching_articles = []

        for article in articles:
            content = article.get("content") or article.get("description", "")

            if content:
                authenticity = classify_article(content)  # 0 = Fake, 1 = True

                # Check if the article matches the selected type
                if (news_type == "True News" and authenticity == 1) or (news_type == "Fake News" and authenticity == 0):
                    matching_articles.append(article)

        # Display results
        if not matching_articles:
            st.warning(f"⚠️ No {news_type.lower()} articles found for '{search_query}'.")
        else:
            for article in matching_articles:
                st.subheader(article["title"])
                st.write(article["description"])
                st.write(f"[Read More]({article['url']})")
                st.write("---")
