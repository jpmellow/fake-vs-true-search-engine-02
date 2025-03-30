import streamlit as st
import pandas as pd
import joblib
import requests
import os
import numpy as np
import time

# Page config
st.set_page_config(
    page_title="Fake vs. True News Search Engine",
    page_icon="📰",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stRadio > div {
        display: flex;
        justify-content: center;
        gap: 2rem;
    }
    div.stButton > button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load and cache the ML models"""
    try:
        with st.spinner('Loading models...'):
            start_time = time.time()
            vectorizer = joblib.load("tfidf_vectorizer.joblib")
            model = joblib.load("news_classifier_model.joblib")
            load_time = time.time() - start_time
            st.sidebar.success(f"✅ Models loaded successfully in {load_time:.2f}s")
            return vectorizer, model
    except Exception as e:
        st.error(f"⚠️ Error loading models: {str(e)}")
        st.stop()

# Load models
try:
    vectorizer, model = load_models()
except Exception as e:
    st.error("⚠️ Failed to load models. Please check the model files.")
    st.stop()

# Get API key from Streamlit secrets or .env
try:
    # Debug info about secrets
    st.sidebar.write("Debug Info:")
    st.sidebar.write("Secrets type:", type(st.secrets))
    st.sidebar.write("Secrets dir:", dir(st.secrets))
    st.sidebar.write("Secrets dict:", dict(st.secrets))
    
    # Try Streamlit secrets first
    if "NEWS_API_KEY" in st.secrets:
        NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
        st.success("✅ Using API key from Streamlit secrets")
    else:
        # Fall back to .env
        from dotenv import load_dotenv
        load_dotenv()
        NEWS_API_KEY = os.getenv("NEWS_API_KEY")
        if NEWS_API_KEY:
            st.success("✅ Using API key from .env file")
        else:
            st.error("⚠️ No API key found")
            st.write("Please either:")
            st.write("1. Add to Streamlit Cloud secrets:")
            st.code('NEWS_API_KEY = "your_api_key_here"')
            st.write("2. Or create a .env file with:")
            st.code('NEWS_API_KEY=your_api_key_here')
            st.stop()
except Exception as e:
    st.error(f"⚠️ Error accessing secrets: {str(e)}")
    st.sidebar.error(f"Debug - Full error: {repr(e)}")
    st.stop()

# Streamlit UI
st.title("📰 Fake vs. True News Search Engine")
st.markdown("---")

def classify_article(content):
    """Classify article content as True (1) or Fake (0)"""
    try:
        if not content:
            return None
            
        # Transform the content
        transformed_content = vectorizer.transform([content])
        
        # Get raw prediction and probabilities
        prediction = model.predict(transformed_content)[0]
        probabilities = model.predict_proba(transformed_content)[0]
        
        # Debug information
        st.sidebar.write("---")
        st.sidebar.write("🔍 Classification Details:")
        st.sidebar.write(f"Raw prediction: {prediction}")
        st.sidebar.write(f"Probabilities: Fake={probabilities[0]:.4f}, True={probabilities[1]:.4f}")
        
        return prediction
        
    except Exception as e:
        st.error(f"⚠️ Classification Error: {str(e)}")
        st.sidebar.error(f"Debug - Error details: {str(e)}")
        return None

# News API configuration
NEWS_API_URL = "https://newsapi.org/v2/everything"

@st.cache_data(ttl=300)  # Cache results for 5 minutes
def fetch_news(query):
    """Fetch news articles from NewsAPI with error handling"""
    try:
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 50,
            "apiKey": NEWS_API_KEY
        }
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get("articles", [])
    except requests.RequestException as e:
        st.error(f"⚠️ API Error: {str(e)}")
        return []

# Search interface
col1, col2 = st.columns([3, 1])
with col1:
    search_query = st.text_input("🔎 Enter a keyword to search for news:", placeholder="e.g., climate change, technology, politics")
with col2:
    news_type = st.radio("📢 Show:", ("True News", "Fake News"), horizontal=True)

# Search button
if st.button("🔍 Search Articles"):
    if not search_query:
        st.warning("⚠️ Please enter a search keyword.")
    else:
        with st.spinner("🔄 Fetching and analyzing news articles..."):
            articles = fetch_news(search_query)
            
            if not articles:
                st.warning(f"⚠️ No articles found for '{search_query}'.")
            else:
                st.info(f"📊 Found {len(articles)} articles to analyze")
                matching_articles = []
                all_classifications = {"True": 0, "Fake": 0}
                
                # Process and classify articles
                progress_bar = st.progress(0)
                for i, article in enumerate(articles):
                    content = article.get("content") or article.get("description", "")
                    if content:
                        authenticity = classify_article(content)
                        if authenticity is not None:
                            if authenticity == 1:
                                all_classifications["True"] += 1
                            else:
                                all_classifications["Fake"] += 1
                                
                            if (news_type == "True News" and authenticity == 1) or \
                               (news_type == "Fake News" and authenticity == 0):
                                matching_articles.append(article)
                    progress_bar.progress((i + 1) / len(articles))
                progress_bar.empty()
                
                # Display classification statistics
                st.write("---")
                st.write("📊 Classification Summary:")
                st.write(f"- True News: {all_classifications['True']} articles")
                st.write(f"- Fake News: {all_classifications['Fake']} articles")
                st.write("---")
                
                # Display results
                if not matching_articles:
                    st.warning(f"⚠️ No {news_type.lower()} found for '{search_query}'.")
                else:
                    st.success(f"🎯 Found {len(matching_articles)} {news_type.lower()}")
                    
                    for article in matching_articles:
                        with st.container():
                            st.markdown(f"### 📰 {article['title']}")
                            st.markdown(f"🔍 {article.get('description', 'No description available.')}")
                            st.markdown(f"🔗 [Read full article]({article['url']})")
                            st.markdown("---")
