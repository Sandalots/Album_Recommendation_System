#!/bin/zsh

# Create venv if neither .venv nor venv exists
if [ ! -d ".venv" ] && [ ! -d "venv" ]; then
  echo "Creating Python virtual environment (.venv)..."
  python3 -m venv .venv
fi

# Activate venv (prefer .venv, fallback to venv)
if [ -d ".venv" ]; then
  source .venv/bin/activate
elif [ -d "venv" ]; then
  source venv/bin/activate
else
  echo "No virtual environment found!" >&2
  exit 1
fi

# Install dependencies
if [ -f requirements.txt ]; then
  echo "Installing dependencies from requirements.txt..."
  pip install --upgrade pip
  pip install -r requirements.txt
fi

python3 error_analyze_recommender.py --no-vizs
