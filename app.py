import streamlit as st
import pandas as pd
import joblib

# Load the trained model and vectorizer
model = joblib.load("news_classifier_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# Streamlit UI
st.title("📰 Fake News Classifier")
st.write("Search for news articles and classify them as **True** or **Fake**.")

# User input for search query
search_query = st.text_input("🔍 Enter a keyword:", "")

# Toggle for True or Fake news
news_type = st.radio("📢 Show:", ("True News", "Fake News"))

# Search button
button = st.button("Search Articles")

if button and search_query:
    try:
        # Load dataset for searching (if available)
        df = pd.read_csv("dataset.csv")
        df_filtered = df[df['article_text'].str.contains(search_query, case=False, na=False)]
        
        if df_filtered.empty:
            st.warning("⚠️ No matching articles found.")
        else:
            # Transform text using TF-IDF
            text_tfidf = vectorizer.transform(df_filtered['article_text'])

            # Predict labels
            predictions = model.predict(text_tfidf)

            # Apply the True/False filter
            label_filter = 1 if news_type == "True News" else 0
            df_filtered = df_filtered[predictions == label_filter]

            # Display results
            if df_filtered.empty:
                st.warning(f"⚠️ No {news_type.lower()} articles found for '{search_query}'.")
            else:
                for _, row in df_filtered.iterrows():
                    st.subheader(row['title'])
                    st.write(row['article_text'][:500] + "...")  # Show preview
                    st.write("---")
    except FileNotFoundError:
        st.error("🚨 Dataset not available. Try a different search or retrain the model.")

