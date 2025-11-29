#!/bin/zsh

# YourNextAlbum Quickstart script: runs all steps after dataset scraping, but skips error_analyzer tasks, we ignore web scraping as it takes too long to run here, 2 hours-ish

# 0. Create venv if neither .venv nor venv exists
if [ ! -d ".venv" ] && [ ! -d "venv" ]; then
  echo "Creating Python virtual environment (.venv)..."
  python3 -m venv .venv
fi

# 1. Activate venv (prefer .venv, fallback to venv)
if [ -d ".venv" ]; then
  source .venv/bin/activate
elif [ -d "venv" ]; then
  source venv/bin/activate
else
  echo "No virtual environment found!" >&2
  exit 1
fi

# 2. Install dependencies for YourNextAlbum
if [ -f requirements.txt ]; then
  echo "Installing dependencies from requirements.txt..."
  pip install --upgrade pip
  pip install -r requirements.txt
fi

# 3. Create album reviews dataset from the scraped data
python3 album_reviews_dataset_creator.py

# 4. Analyze sentiment of album reviews
python3 album_reviews_text_sentiment_analyser.py

# 5. Build album recommender models
python3 album_recommender_model.py

# Notify user of core YourNextAlbum setup completion
echo "Quickstart complete: venv ready, dependencies installed, dataset created, sentiment analyzed, recommender built."

# 6. Run the Streamlit YourNextAlbum app
streamlit run album_recommender_prompt_app.py