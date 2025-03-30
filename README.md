# Fake vs. True News Search Engine

A Streamlit-based web application that combines machine learning classification with an intelligent news source recommendation system. The app classifies news articles as either true or fake using an XGBoost model and recommends reliable news sources based on classification patterns.

## Live Demo
The app is deployed on Streamlit Cloud and will remain accessible for review. [Click here to try the app](https://fake-vs-true-search-engine-02.streamlit.app/)

## Features
- Real-time news article fetching using NewsAPI
- Machine learning-based classification (XGBoost)
- TF-IDF text vectorization for article analysis
- Intelligent news source recommendation system
- Detailed classification probabilities
- Clean and intuitive user interface
- Support for any news topic search
- Secure API key input (no configuration needed)

## System Components

### 1. News Classification System
- Uses TF-IDF vectorization to convert article text into numerical features
- Implements XGBoost classifier trained on a balanced dataset of true/fake news
- Achieves high accuracy in distinguishing true from fake news
- Performance metrics:
  - Training accuracy: 97.66%
  - Testing accuracy: 97.56%
  - Balanced precision and recall

### 2. Source Recommendation System
Our recommendation system analyzes news sources and suggests similar sources based on their reliability patterns. Instead of using traditional embedding models, we implemented a novel approach that:

1. **Vector Representation:**
   - Creates vectors from source reliability patterns
   - Considers true/fake classification ratios
   - Incorporates confidence levels based on sample size

2. **Reliability Scoring:**
   - Calculates weighted reliability scores for each news source
   - Implements confidence levels:
     - High: 5+ articles analyzed
     - Medium: 3-4 articles analyzed
     - Low: 1-2 articles analyzed
   - Adjusts scores towards neutral (50%) for low-confidence sources

3. **Similarity Search:**
   - Uses cosine similarity between source vectors
   - Considers both reliability patterns and classification results
   - Weights recommendations based on:
     - 70% reliability score
     - 30% similarity to current sources

4. **Recommendation Logic:**
   - For True News: Recommends sources with >40% reliability
   - For Fake News: Recommends sources with <60% reliability
   - Prioritizes sources with higher confidence levels
   - Dynamically updates based on real-time classifications

## Technical Stack
- Python 3.8+
- Streamlit 1.31.1
- XGBoost 2.0.3
- scikit-learn 1.6.1
- pandas 2.2.0
- NewsAPI

## Setup and Installation

### Prerequisites
1. Python 3.8 or higher
2. NewsAPI key (get it from [https://newsapi.org](https://newsapi.org))

### Using the Live Demo
1. Visit the [live demo](https://fake-vs-true-search-engine-02.streamlit.app/)
2. Enter your NewsAPI key in the sidebar (it will be securely hidden)
3. Start searching and analyzing news articles!

### Local Installation
1. Clone the repository:
   ```bash
   git clone [your-repo-url]
   cd fake-vs-true-search-engine-01
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root (optional, you can also enter the key in the app):
   ```
   NEWS_API_KEY=your_api_key_here
   ```

4. Train the model (optional, pre-trained model included):
   ```bash
   python train_model.py
   ```

5. Run the app:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Enter your NewsAPI key in the sidebar (first time only)
2. Enter a search term in the search box
3. Select whether to view True or Fake news classifications
4. Click "Search Articles" to fetch and analyze news
5. View article classifications and source recommendations

## Project Structure
```
├── app.py              # Main Streamlit application
├── train_model.py      # Model training script
├── requirements.txt    # Project dependencies
├── .env               # Environment variables (optional)
├── tfidf_vectorizer.joblib  # Trained vectorizer
└── news_classifier_model.joblib  # Trained model
```

## Contributing
Feel free to submit issues and enhancement requests!

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- NewsAPI for providing access to real-time news data
- Streamlit for the excellent web app framework
- The Fake News Dataset used for training