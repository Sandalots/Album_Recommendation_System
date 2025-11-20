# Workspace Overview

This project is an AI-powered album recommendation system built around Pitchfork review data. It consists of several main components:

- **album_scraper.py**: Scrapes thousands of album reviews from Pitchfork using Selenium and BeautifulSoup. The scraper collects review text, scores, album/artist info, and album art URLs.
-- **dataset_creator.py**: Cleans and preprocesses the raw review data using NLTK (Natural Language Toolkit) for tokenization, stopword removal, lemmatization, and stemming, plus HTML/text cleaning and feature engineering (e.g., sentiment, mood, instrumentation, context).
- **sentiment_analyzer.py**: Uses NLP models (HuggingFace Transformers) to analyze review sentiment, summarize text, and extract features like mood, energy, instrumentation, production style, novelty, and more. Models used include DistilBERT for sentiment and BART for summarization.
- **recommender_model.py**: Builds recommendation models using TF-IDF and SentenceTransformer embeddings. It combines review features, album metadata, and semantic search to match user prompts to relevant albums.
- **app.py**: Streamlit web app for interactive recommendations. Users enter a prompt (e.g., “dreamy synthpop for studying”), and the app displays recommended albums with cover art, artist, score, and key highlights. The UI features custom CSS for a modern look and smooth animations.

## Data Pipeline

1. **Scraping**: `album_scraper.py` collects thousands of reviews from Pitchfork, saving them as CSV/JSON.
2. **Preprocessing**: `dataset_creator.py` cleans and normalizes the data, engineering features for downstream analysis.
3. **Analysis**: `sentiment_analyzer.py` applies NLP models and custom keyword lists to extract rich features from each review.
4. **Recommendation**: `recommender_model.py` builds semantic and keyword-based models to match user queries to albums.
5. **User Interface**: `app.py` provides a fast, interactive Streamlit UI for exploring recommendations.

## Models & Features

- **Sentiment Analysis**: DistilBERT (HuggingFace) for positive/negative/neutral sentiment.
- **Summarization**: BART (HuggingFace) for concise review summaries.
- **Semantic Search**: SentenceTransformer for embedding-based matching.
- **TF-IDF**: For keyword-based similarity.
- **Feature Extraction**: Custom logic for mood, energy, instrumentation, production, novelty, context, and more.

## Streamlit App

- Enter a prompt describing your desired album (genre, mood, context, etc.).
- View recommended albums with cover art, artist, score, and highlights.
- UI features include centered content, fade-in animations, custom button styling, and hidden Streamlit menu/deploy buttons.

# 🎵 Album Recommendation AI System
AI-powered album recommender using Pitchfork reviews with advanced NLP, sentiment analysis, and semantic search.

## Quick Start
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python album_scraper.py          # Scrape reviews
python dataset_creator.py        # Preprocess with NLTK and turn into a dataset
python sentiment_analyzer.py     # Analyze (8+ feature types) from review texts
python recommender_model.py      # builds the model and recommends albums based on given prompt
streamlit run app.py             # Launches a streamlit UI to allow users to prompt the created model for album recommendations
```

## Analysis Features
- **Sentiment & Consensus** - Tone + score alignment
- **Instrumentation** - Guitar, synth, drums, vocals, etc.
- **Mood & Energy** - Sad/happy/dark + high/medium/low
- **Production** - Polished/raw/experimental, analog/digital
- **Novelty** - Innovative vs. derivative detection
- **Context** - Party, study, workout, relaxation, etc.
- **Temporal** - Vintage/contemporary/timeless
- **Polarizing** - Divisive "love it or hate it" albums
