import streamlit as st
import pandas as pd
import joblib
import requests
import os
import numpy as np
import time
from urllib.parse import urlparse
from collections import defaultdict
import re

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

def extract_base_url(url):
    """Extract the base domain from a URL."""
    try:
        parsed = urlparse(url)
        # Remove www. if present
        domain = re.sub(r'^www\.', '', parsed.netloc)
        return domain
    except:
        return None

def analyze_sources(articles, authenticity_results):
    """Analyze news sources and their reliability patterns."""
    source_stats = defaultdict(lambda: {'true': 0, 'fake': 0, 'total': 0, 'articles': []})
    
    for article, is_true in zip(articles, authenticity_results):
        if not article.get('url'):
            continue
            
        base_url = extract_base_url(article['url'])
        if not base_url:
            continue
            
        source_stats[base_url]['total'] += 1
        source_stats[base_url]['true' if is_true == 1 else 'fake'] += 1
        source_stats[base_url]['articles'].append(article)
        
    return source_stats

def get_source_reliability_score(stats):
    """Calculate a reliability score for a source."""
    if stats['total'] == 0:
        return 0
    return (stats['true'] / stats['total']) * 100

def find_similar_sources(source_stats, current_sources, news_type, top_n=5):
    """Find similar sources based on reliability patterns and news type preference."""
    # Convert reliability patterns to vectors
    source_vectors = {}
    reliability_scores = {}
    
    for source, stats in source_stats.items():
        total = stats['total']
        if total > 0:
            reliability = get_source_reliability_score(stats)
            reliability_scores[source] = reliability
            true_ratio = stats['true'] / total
            fake_ratio = stats['fake'] / total
            source_vectors[source] = np.array([true_ratio, fake_ratio])
    
    # Find similar sources for each current source
    similar_sources = []
    current_sources = set(current_sources)
    
    # Calculate average reliability of current sources
    current_reliabilities = [reliability_scores[s] for s in current_sources if s in reliability_scores]
    target_reliability = np.mean(current_reliabilities) if current_reliabilities else 50.0
    
    # Get all potential sources excluding current ones
    potential_sources = set(source_vectors.keys()) - current_sources
    
    # Score each potential source
    source_scores = []
    for source in potential_sources:
        reliability = reliability_scores[source]
        
        # Skip sources with low reliability when searching for true news
        if news_type == "True News" and reliability < 50:
            continue
        # Skip sources with high reliability when searching for fake news
        if news_type == "Fake News" and reliability > 50:
            continue
            
        # Calculate similarity score
        similarity_scores = []
        for curr_source in current_sources:
            if curr_source in source_vectors:
                similarity = np.dot(source_vectors[curr_source], source_vectors[source])
                similarity_scores.append(similarity)
        
        # Average similarity with current sources
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
        
        # Combine similarity and reliability for final score
        if news_type == "True News":
            final_score = (avg_similarity * 0.3) + (reliability * 0.7)
        else:
            final_score = (avg_similarity * 0.3) + ((100 - reliability) * 0.7)
            
        source_scores.append((final_score, reliability, source))
    
    # Sort by final score and get top recommendations
    source_scores.sort(reverse=True)
    similar_sources = [(s[2], s[1]) for s in source_scores[:top_n]]
    
    return similar_sources

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

# Get API key from .env or user input
try:
    # Try .env file first (for local development)
    from dotenv import load_dotenv
    load_dotenv()
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    
    # If no .env, ask for API key input
    if not NEWS_API_KEY:
        st.sidebar.markdown("### 🔑 API Key Configuration")
        input_api_key = st.sidebar.text_input(
            "Enter your NewsAPI key:",
            type="password",
            help="Get your key at https://newsapi.org"
        )
        if input_api_key:
            NEWS_API_KEY = input_api_key
            st.sidebar.success("✅ API key set")
        else:
            st.error("⚠️ Please enter your NewsAPI key in the sidebar")
            st.write("You can get a free API key at [NewsAPI.org](https://newsapi.org)")
            st.stop()
    else:
        st.sidebar.success("✅ Using API key from .env file")
        
except Exception as e:
    st.error(f"⚠️ Error accessing API key: {str(e)}")
    st.stop()

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
                authenticity_results = []
                
                # Process and classify articles
                progress_bar = st.progress(0)
                for i, article in enumerate(articles):
                    content = article.get("content") or article.get("description", "")
                    if content:
                        authenticity = classify_article(content)
                        if authenticity is not None:
                            authenticity_results.append(authenticity)
                            if authenticity == 1:
                                all_classifications["True"] += 1
                            else:
                                all_classifications["Fake"] += 1
                                
                            if (news_type == "True News" and authenticity == 1) or \
                               (news_type == "Fake News" and authenticity == 0):
                                matching_articles.append(article)
                    progress_bar.progress((i + 1) / len(articles))
                progress_bar.empty()
                
                # Create two columns for results and recommendations
                col1, col2 = st.columns([2, 1])
                
                with col1:
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
                
                with col2:
                    # Analyze and display source recommendations
                    source_stats = analyze_sources(articles, authenticity_results)
                    current_sources = [extract_base_url(a['url']) for a in matching_articles if a.get('url')]
                    similar_sources = find_similar_sources(source_stats, current_sources, news_type)
                    
                    st.markdown("### 📈 News Source Analysis")
                    
                    # Show current sources in collapsed expander
                    if current_sources:
                        with st.expander("🔍 Current Sources", expanded=False):
                            for source in set(current_sources):
                                stats = source_stats[source]
                                reliability = get_source_reliability_score(stats)
                                st.markdown(f"- **{source}**")
                                st.markdown(f"  - Reliability: {reliability:.1f}%")
                                st.markdown(f"  - Articles: {stats['total']}")
                    
                    # Show recommended sources in expanded expander
                    with st.expander("🌟 Recommended Sources", expanded=True):
                        if not similar_sources:
                            st.info(f"No additional {news_type.lower()} sources found with suitable reliability scores.")
                        else:
                            for source, reliability in similar_sources:
                                stats = source_stats[source]
                                st.markdown(f"- **{source}**")
                                st.markdown(f"  - Reliability: {reliability:.1f}%")
                                st.markdown(f"  - Articles: {stats['total']}")
