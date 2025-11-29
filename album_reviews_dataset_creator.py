import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import string
import html
import os


# Ensure NLTK resources are available
def ensure_nltk_resource(resource):
    try:
        nltk.data.find(resource)

    except LookupError:
        nltk.download(resource.split('/')[-1], quiet=True)


ensure_nltk_resource('tokenizers/punkt')
ensure_nltk_resource('corpora/stopwords')
ensure_nltk_resource('corpora/wordnet')
ensure_nltk_resource('taggers/averaged_perceptron_tagger')

print("=== Scrapped Album Reviews Dataset Preprocessing ===\n")

df = pd.read_csv('outputs/pitchfork_reviews_raw.csv')

# Fill missing artist names with 'Various Artists'
df['artist_name'] = df['artist_name'].fillna('Various Artists')

print(f"Input Scrapped Pitchfork Dataset shape: {df.shape}")

missing_values = df.isnull().sum()
print(f"Missing values in Pitchfork Dataset: {missing_values.sum()}\n")

print("Step 1: Basic Missing Values Cleaning")

df['genre'] = df['genre'].fillna('Unknown')
mean_score = df['score'].astype(float).mean()

df['score'] = df['score'].fillna(mean_score)
median_year = df['release_year'].astype(float).median()

df['release_year'] = df['release_year'].fillna(median_year)
df['review_text'] = df['review_text'].fillna('')

print("Filled missing values.\n")

print("Step 2: Text Preprocessing & Data Normalization")

# Initialize text processing tools
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Keep music-specific words that might be in stopwords
music_stopwords = stop_words - {'not', 'no', 'more', 'most', 'very', 'only', 'too', 'just'}

def preprocess_text(text):
    """Comprehensive text preprocessing"""
    if pd.isna(text) or not text:
        return '', '', ''

    # First: Decode HTML entities (e.g., &amp; -> &, &#39; -> ')
    text = html.unescape(text)

    # Add space before capital letters that might be conjoined (e.g., "theBlack" -> "the Black")
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # Convert to lowercase
    text_lower = text.lower()

    # Remove URLs
    text_clean = re.sub(r'http\S+|www\S+|https\S+', '', text_lower, flags=re.MULTILINE)

    # Remove email addresses
    text_clean = re.sub(r'\S+@\S+', '', text_clean)

    # Remove special characters but keep apostrophes and hyphens in words
    text_clean = re.sub(r'[^\w\s\'-]', ' ', text_clean)

    # Remove extra whitespace
    text_clean = re.sub(r'\s+', ' ', text_clean).strip()

    # Tokenize
    tokens = word_tokenize(text_clean)

    # Remove stopwords and punctuation
    tokens_filtered = [ w for w in tokens if w not in music_stopwords and w not in string.punctuation and len(w) > 2]

    # Lemmatization (preserves meaning better)
    tokens_lemmatized = [lemmatizer.lemmatize(w) for w in tokens_filtered]

    # Stemming (more aggressive, shorter words)
    tokens_stemmed = [stemmer.stem(w) for w in tokens_filtered]

    # Join back into text
    processed_text = ' '.join(tokens_lemmatized)
    stemmed_text = ' '.join(tokens_stemmed)

    return processed_text, stemmed_text, text_clean


print("Processing album review texts...")

# Apply preprocessing
text_processing = df['review_text'].apply(preprocess_text)

df['review_text_processed'] = text_processing.apply(lambda x: x[0])
df['review_text_stemmed'] = text_processing.apply(lambda x: x[1])
df['review_text_clean'] = text_processing.apply(lambda x: x[2])

print("Processed text columns created.\n")

print("="*80)
print("STEP 3: Advanced Feature Engineering")
print("="*80)

# Text-based features
df['review_length'] = df['review_text'].str.len()
df['word_count'] = df['review_text'].str.split().str.len()
df['processed_word_count'] = df['review_text_processed'].str.split().str.len()
df['unique_word_ratio'] = df.apply(lambda row: len(set(row['review_text_processed'].split())) / max(row['processed_word_count'], 1), axis=1)

# Sentiment indicators from text
df['exclamation_count'] = df['review_text'].str.count('!')
df['question_count'] = df['review_text'].str.count(r'\?')
df['avg_word_length'] = df['review_text_processed'].apply(lambda x: sum(len(word) for word in x.split()) / max(len(x.split()), 1) if x else 0)

# Score normalization and categorization
df['score_normalized'] = (df['score'].astype(float) - df['score'].astype(float).min()) / (df['score'].astype(float).max() - df['score'].astype(float).min())

df['score_category'] = pd.cut(df['score'].astype(float), bins=[0, 5, 7, 8.5, 10], labels=['Poor', 'Good', 'Great', 'Masterpiece'])

# Temporal features
df['decade'] = (df['release_year'].astype(float) // 10 * 10).astype(int)
df['release_year_normalized'] = (df['release_year'].astype(float) - df['release_year'].astype(float).min()) / (df['release_year'].astype(float).max() - df['release_year'].astype(float).min())

# Genre standardization
df['primary_genre'] = df['genre'].str.split('/').str[0].str.strip().str.lower()
df['genre_count'] = df['genre'].str.count('/') + 1

# Artist/Album name features
df['artist_name_clean'] = df['artist_name'].str.lower().str.strip()
df['album_name_clean'] = df['album_name'].str.lower().str.strip()
df['album_title_length'] = df['album_name'].str.len()

print("Advanced album review features created.\n")

print("="*80)
print("STEP 4: Data Quality Checks & Outlier Detection")
print("="*80)

duplicates = df.duplicated(subset=['album_name', 'artist_name']).sum()
print(f"Duplicate albums: {duplicates}")

empty_reviews = (df['review_text'].str.len() < 100).sum()
print(f"Very short reviews (<100 chars): {empty_reviews}")

# Outlier detection for review length
q1_length = df['review_length'].quantile(0.25)
q3_length = df['review_length'].quantile(0.75)
iqr_length = q3_length - q1_length
outliers_length = ((df['review_length'] < (q1_length - 1.5 * iqr_length)) | (df['review_length'] > (q3_length + 1.5 * iqr_length))).sum()
print(f"Review length outliers: {outliers_length}")

# Check for anomalous scores
anomalous_scores = ((df['score'].astype(float) < 0) | (df['score'].astype(float) > 10)).sum()
print(f"Anomalous scores (outside 0-10 range): {anomalous_scores}")

missing_final = df.isnull().sum().sum()
print(f"Total missing values after processing: {missing_final}\n")

print("="*80)
print("STEP 5: Dataset Statistics & Validation")
print("="*80)

print(f"\nScore Distribution:")
print(df['score_category'].value_counts().sort_index())

print(f"\nTop 10 Genres:")
print(df['primary_genre'].value_counts().head(10))

print(f"\nReview Text Stats:")

print(f"  Original:")
print(f"    Average length: {df['review_length'].mean():.0f} characters")
print(f"    Average words: {df['word_count'].mean():.0f} words")
print(f"    Min length: {df['review_length'].min()}")
print(f"    Max length: {df['review_length'].max()}")

print(f"  Processed:")
print(f"    Average words: {df['processed_word_count'].mean():.0f} words")
print(f"    Average unique word ratio: {df['unique_word_ratio'].mean():.2f}")
print(
    f"    Average word length: {df['avg_word_length'].mean():.2f} characters")

print(f"\n" + "="*80)
print("Saving preprocessed pitchfork reviews dataset...")
print("="*80)

# Ensure outputs directory exists
os.makedirs('outputs', exist_ok=True)

df.to_csv('outputs/pitchfork_reviews_preprocessed.csv', index=False)
print(" Saved to: outputs/pitchfork_reviews_preprocessed.csv")

print(f"\nFinal dataset shape: {df.shape}")
print(f"Total columns: {len(df.columns)}")
print(f"\nNew columns added:")

new_cols = ['review_text_clean', 'review_text_processed', 'review_text_stemmed',
            'processed_word_count', 'unique_word_ratio', 'exclamation_count',
            'question_count', 'avg_word_length', 'score_normalized',
            'release_year_normalized', 'genre_count', 'artist_name_clean',
            'album_name_clean', 'album_title_length']

for col in new_cols:
    if col in df.columns:
        print(f"  - {col}")

print()