import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
import re

# Load datasets
print("Loading datasets...")
true_df = pd.read_csv('../Fake-News-Dataset/True.csv')
fake_df = pd.read_csv('../Fake-News-Dataset/Fake.csv')

# Function to clean text
def clean_text(text):
    # Remove source markers like "WASHINGTON (Reuters) -"
    text = re.sub(r'\([^)]*\)', '', text)
    # Remove common news source identifiers
    text = re.sub(r'\b(Reuters|AP|AFP|BBC)\b', '', text)
    # Remove locations often used in datelines
    text = re.sub(r'(WASHINGTON|NEW YORK|LONDON) -', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

# Add labels
true_df['label'] = 1  # True news
fake_df['label'] = 0  # Fake news

# Clean the text
print("Cleaning text...")
true_df['clean_title'] = true_df['title'].apply(clean_text)
true_df['clean_text'] = true_df['text'].apply(clean_text)
fake_df['clean_title'] = fake_df['title'].apply(clean_text)
fake_df['clean_text'] = fake_df['text'].apply(clean_text)

# Balance datasets
min_size = min(len(true_df), len(fake_df))
true_df = true_df.sample(n=min_size, random_state=42)
fake_df = fake_df.sample(n=min_size, random_state=42)

# Combine datasets
print("Combining datasets...")
df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Prepare text data (combine cleaned title and text)
print("Preparing text data...")
df['content'] = df['clean_title'] + " " + df['clean_text']

# Split into features and labels
X = df['content']
y = df['label']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create and fit TF-IDF vectorizer
print("Creating and fitting TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95  # Remove very common terms
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train XGBoost model with adjusted parameters
print("Training XGBoost model...")
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,  # Balanced dataset
    random_state=42
)
model.fit(X_train_tfidf, y_train)

# Evaluate model
print("\nModel Evaluation:")
train_score = model.score(X_train_tfidf, y_train)
test_score = model.score(X_test_tfidf, y_test)
print(f"Training accuracy: {train_score:.4f}")
print(f"Testing accuracy: {test_score:.4f}")

# Print feature importances
feature_names = vectorizer.get_feature_names_out()
importances = model.feature_importances_
top_features = sorted(zip(importances, feature_names), reverse=True)[:20]
print("\nTop 20 important features:")
for importance, feature in top_features:
    print(f"{feature}: {importance:.4f}")

# Save model and vectorizer
print("\nSaving model and vectorizer...")
joblib.dump(model, 'news_classifier_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

print("Done! Model and vectorizer have been saved.")
