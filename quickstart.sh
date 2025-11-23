#!/bin/zsh
# Quickstart script: runs all steps after dataset scraping, but skips error_analyzer tasks

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

# 2. Install dependencies
if [ -f requirements.txt ]; then
  echo "Installing dependencies from requirements.txt..."
  pip install --upgrade pip
  pip install -r requirements.txt
fi

# 3. Create dataset (assuming dataset_creator.py is the script for this)
python3 dataset_creator.py

# 4. Run sentiment analysis (if applicable)
python3 sentiment_analyzer.py

# 5. Build or enhance the recommender (if needed)
python3 enhanced_recommender.py

# 6. (Optional) Any other post-processing or preparation steps can be added here

echo "Quickstart complete: venv ready, dependencies installed, dataset created, sentiment analyzed, recommender built."

streamlit run app.py
