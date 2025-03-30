# Fake vs. True News Search Engine

A Streamlit-based web application that uses machine learning to classify news articles as either true or fake. The app fetches real-time news articles based on user queries and analyzes them using a trained XGBoost model.

## Live Demo
[Click here to try the app on Streamlit Cloud](#) (Link will be added after deployment)

## Features
- Real-time news article fetching using NewsAPI
- Machine learning-based classification (XGBoost)
- TF-IDF text vectorization
- Detailed classification probabilities
- Clean and intuitive user interface
- Support for any news topic search

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

3. Create a `.env` file in the project root:
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

## Dataset
The model was trained on a balanced dataset of true and fake news articles, with features including:
- Article text content
- Writing style and tone
- Semantic patterns
- Source credibility markers

## Model Performance
- Training accuracy: 97.66%
- Testing accuracy: 97.56%
- Balanced precision and recall for both true and fake news classes

## Usage
1. Enter a search term in the search box
2. Select whether to view True or Fake news classifications
3. Click "Search Articles" to fetch and analyze news
4. View the results with classification probabilities

## Project Structure
```
├── app.py              # Main Streamlit application
├── train_model.py      # Model training script
├── requirements.txt    # Project dependencies
├── .env               # Environment variables (not in repo)
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