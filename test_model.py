import pandas as pd
import joblib
import numpy as np

# Load the model and vectorizer
print("Loading model and vectorizer...")
vectorizer = joblib.load('tfidf_vectorizer.joblib')
model = joblib.load('news_classifier_model.joblib')

# Load some examples from both datasets
print("\nLoading test examples...")
true_df = pd.read_csv('../Fake-News-Dataset/True.csv')
fake_df = pd.read_csv('../Fake-News-Dataset/Fake.csv')

# Function to test classification
def test_article(content, expected_label):
    transformed = vectorizer.transform([content])
    prediction = model.predict(transformed)[0]
    probs = model.predict_proba(transformed)[0]
    return prediction, probs

# Test some true news articles
print("\nTesting TRUE news articles:")
for i in range(5):
    content = true_df.iloc[i]['title'] + " " + true_df.iloc[i]['text']
    pred, probs = test_article(content, 1)
    print(f"\nArticle {i+1}:")
    print(f"Title: {true_df.iloc[i]['title'][:100]}...")
    print(f"Prediction: {'True' if pred == 1 else 'Fake'}")
    print(f"Probabilities: Fake={probs[0]:.4f}, True={probs[1]:.4f}")

# Test some fake news articles
print("\nTesting FAKE news articles:")
for i in range(5):
    content = fake_df.iloc[i]['title'] + " " + fake_df.iloc[i]['text']
    pred, probs = test_article(content, 0)
    print(f"\nArticle {i+1}:")
    print(f"Title: {fake_df.iloc[i]['title'][:100]}...")
    print(f"Prediction: {'True' if pred == 1 else 'Fake'}")
    print(f"Probabilities: Fake={probs[0]:.4f}, True={probs[1]:.4f}")

# Print model details
print("\nModel Details:")
print(f"Number of features: {len(vectorizer.get_feature_names_out())}")
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    feature_names = vectorizer.get_feature_names_out()
    top_features = sorted(zip(importances, feature_names), reverse=True)[:10]
    print("\nTop 10 important features:")
    for importance, feature in top_features:
        print(f"{feature}: {importance:.4f}")
