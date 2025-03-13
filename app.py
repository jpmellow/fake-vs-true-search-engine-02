import streamlit as st
import pandas as pd
import joblib
import requests
import os
from dotenv import load_dotenv  # Load environment variables

# Load environment variables from .env
load_dotenv()

# Get API key from .env
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
if not NEWS_API_KEY:
    st.error("⚠️ API key is missing. Please add it to your .env file.")

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
