import streamlit as st
import pandas as pd
import joblib
import requests
import os
from dotenv import load_dotenv
import numpy as np

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

# Load trained classifier and vectorizer
try:
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    if not hasattr(vectorizer, 'idf_') or vectorizer.idf_ is None:
        st.error("⚠️ TF-IDF vectorizer is not properly fitted. Please ensure the vectorizer was trained before saving.")
        st.stop()
    model = joblib.load("news_classifier_model.joblib")
    st.sidebar.success("✅ Models loaded successfully")
    
    # Debug information about the model
    st.sidebar.write("### Model Information")
    st.sidebar.write(f"Vectorizer features: {len(vectorizer.get_feature_names_out())}")
    st.sidebar.write(f"Model type: {type(model).__name__}")
    
except FileNotFoundError as e:
    st.error("⚠️ Required model files not found. Please ensure both .joblib files are present.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Error loading models: {str(e)}")
    st.stop()

# Load environment variables from .env
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Streamlit UI
st.title("📰 Fake vs. True News Search Engine")
st.markdown("---")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This app uses machine learning to classify news articles as True or Fake.
    
    - Enter a keyword to search for news
    - Articles are fetched in real-time
    - Each article is classified using our trained model
    - Toggle between True and Fake news results
    """)
    
    st.markdown("---")
    st.markdown("### 🛠️ Powered by:")
    st.markdown("- NewsAPI")
    st.markdown("- XGBoost Classifier")
    st.markdown("- TF-IDF Vectorization")

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
        st.sidebar.write(f"Probabilities: {probabilities}")
        st.sidebar.write(f"Feature vector shape: {transformed_content.shape}")
        
        # Sample some important features
        feature_names = vectorizer.get_feature_names_out()
        feature_importances = np.zeros(len(feature_names))
        for i, score in enumerate(transformed_content.toarray()[0]):
            if score > 0:
                feature_importances[i] = score
        
        # Show top features for this text
        top_features_idx = np.argsort(feature_importances)[-5:]
        st.sidebar.write("\nTop features in text:")
        for idx in top_features_idx:
            if feature_importances[idx] > 0:
                st.sidebar.write(f"- {feature_names[idx]}: {feature_importances[idx]:.4f}")
        
        return prediction
        
    except Exception as e:
        st.error(f"⚠️ Classification Error: {str(e)}")
        st.sidebar.error(f"Debug - Error details: {str(e)}")
        return None

# News API configuration
NEWS_API_URL = "https://newsapi.org/v2/everything"

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

# Main content
if not NEWS_API_KEY:
    st.error("⚠️ API key is missing. Please add NEWS_API_KEY to your `.env` file.")
    st.stop()

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
                for article in articles:
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
