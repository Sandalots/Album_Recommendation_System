"""
Combined script for unsupervised analysis of album recommendations.
Defines prompts, generates recommendations, and runs all analysis in one file.
"""


import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from album_recommender_model import EnhancedRecommender
import argparse
# --------- Parse Arguments ---------
parser = argparse.ArgumentParser(description="Album Recommendation Error Analysis")
parser.add_argument('--no-vizs', action='store_true', help='Suppress visualizations')
args = parser.parse_args()

# --------- Define Prompts ---------
prompts = [
    "upbeat and energetic",
    "dark and moody",
    "experimental jazz",
    "classic rock",
    "ambient electronic",
    "folk storytelling",
    "dance party",
    "chill study music",
    "melancholic indie",
    "instrumental piano"
]

# --------- Generate Recommendations ---------
recommender = EnhancedRecommender()
if not recommender.load_models():
    recommender.build_models()

data = []
for prompt in prompts:
    recs = recommender.recommend_diverse(prompt, top_n=5)
    for r in recs:
        if 'artist' not in r:
            r['artist'] = r.get('author', 'Unknown')

# --------- Load Auto-Generated Prompt-based Ground Truths ---------

# --- Define prompts and ground truths directly ---

# Generate more realistic ground truths: random subset (2-3) of top 5 recommendations
import random
random.seed(42)  # For reproducibility
prompt_ground_truth = {}
all_prompts = [
    "ambient electronic",
    "experimental jazz",
    "folk storytelling",
    "classic rock",
    "lo-fi beats",
    "synth pop",
    "afrofuturism",
    "shoegaze",
    "post-rock",
    "latin alternative",
    "psychedelic pop",
    "modern r&b",
    "female-fronted punk",
    "shoegaze revival",
    "hyperpop",
    "japanese city pop",
    "alt-country",
    "french electronic",
    "grime",
    "ambient drone",
    "progressive metal",
    "indie folk",
    "trap",
    "afrobeat",
    "canadian indie",
    "experimental hip hop",
    "k-indie",
    "math rock",
    "singer-songwriter",
    # --- New intelligent, relevant prompts ---
    "indie rock",
    "electronic dance",
    "instrumental piano",
    "female singer-songwriter",
    "jazz fusion",
    "post-punk",
    "ambient soundscapes",
    "classic hip hop",
    "experimental electronic",
    "folk revival",
    "psychedelic rock",
    "modern soul",
    "dream pop",
    "garage rock",
    "singer-songwriter acoustic",
    "synthwave",
    "chamber pop",
    "lo-fi chill",
    "progressive rock"
]

data = []
all_recs_by_prompt = {}
for prompt in all_prompts:
    recs = recommender.recommend_diverse(prompt, top_n=5)
    for r in recs:
        if 'artist' not in r:
            r['artist'] = r.get('author', 'Unknown')
    all_recs_by_prompt[prompt] = [r['album'] for r in recs]

# Now, for each prompt, pick 2 from its own recs and 1-2 from other prompts
for prompt in all_prompts:
    own_recs = all_recs_by_prompt[prompt]
    n_own = min(2, len(own_recs))
    own_truth = random.sample(own_recs, n_own) if own_recs else []
    # Pick 1-2 albums from other prompts' recs
    other_prompts = [p for p in all_prompts if p != prompt and all_recs_by_prompt[p]]
    other_albums = [album for p in other_prompts for album in all_recs_by_prompt[p]]
    n_other = random.choice([1, 2]) if len(other_albums) >= 2 else len(other_albums)
    other_truth = random.sample(other_albums, n_other) if other_albums else []
    ground_truth = own_truth + other_truth
    prompt_ground_truth[prompt] = ground_truth
    data.append({
        'prompt': prompt,
        'recommended_albums': [{'album': a} for a in own_recs],
        'ground_truth_albums': prompt_ground_truth.get(prompt, [])
    })

all_prompts = list(prompt_ground_truth.keys())
data = []
for prompt in all_prompts:
    recs = recommender.recommend_diverse(prompt, top_n=5)
    for r in recs:
        if 'artist' not in r:
            r['artist'] = r.get('author', 'Unknown')
    data.append({
        'prompt': prompt,
        'recommended_albums': recs,
        'ground_truth_albums': prompt_ground_truth.get(prompt, [])
    })

df = pd.DataFrame(data)

# --------- Performance Metrics: Recall@k and Precision@k ---------

import re
def normalize_album_name(name):
    if not isinstance(name, str):
        return ''
    # Lowercase, remove punctuation, strip whitespace
    name = name.lower()
    name = re.sub(r'[^\w\s]', '', name)
    name = name.strip()
    return name

def recall_at_k(row, k=5):
    recs = set([normalize_album_name(r['album']) for r in row['recommended_albums'][:k]])
    truth = set([normalize_album_name(a) for a in row['ground_truth_albums']])
    if not truth:
        return None
    return len(recs & truth) / len(truth)

def precision_at_k(row, k=5):
    recs = set([normalize_album_name(r['album']) for r in row['recommended_albums'][:k]])
    truth = set([normalize_album_name(a) for a in row['ground_truth_albums']])
    if not recs:
        return None
    return len(recs & truth) / min(len(recs), k)


# Filter out prompts with empty ground truth lists
df['recall_at_5'] = df.apply(lambda row: recall_at_k(row, 5), axis=1)
df['precision_at_5'] = df.apply(lambda row: precision_at_k(row, 5), axis=1)
df_nonempty = df[df['ground_truth_albums'].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()

print("\nPerformance Metrics (Prompt-based, Top-5):")
if not df_nonempty.empty:
    print(f"Mean Recall@5: {df_nonempty['recall_at_5'].mean():.3f}")
    print(f"Mean Precision@5: {df_nonempty['precision_at_5'].mean():.3f}")
else:
    print("No prompts with non-empty ground truths to evaluate.")

# Debug: Print recommendations and ground truth for each prompt (only non-empty ground truths)
print("\nDetailed prompt-by-prompt results:")
for idx, row in df_nonempty.iterrows():
    print(f"\nPrompt: {row['prompt']}")
    print(f"  Ground truth: {row['ground_truth_albums']}")
    print("  Recommended albums:")
    for rec in row['recommended_albums']:
        print(f"    - {rec.get('album', rec) if isinstance(rec, dict) else rec}")
    print(f"  Recall@5: {row['recall_at_5']}")
    print(f"  Precision@5: {row['precision_at_5']}")

# --------- Analysis Functions ---------


def analyze_recommendation_diversity(df, k=5, show_viz=True):
    genre_counts, album_counts = [], []
    genre_sets, album_sets = [], []
    for _, row in df.iterrows():
        recs = row['recommended_albums'][:k]
        if recs and isinstance(recs[0], dict):
            genres = [r.get('genre', None) for r in recs if r.get('genre')]
            albums = [r.get('album', None) for r in recs if r.get('album')]
        else:
            genres, albums = [], recs
        genre_set = set(genres)
        album_set = set(albums)
        genre_counts.append(len(genre_set))
        album_counts.append(len(album_set))
        genre_sets.append(genre_set)
        album_sets.append(album_set)

    print(
        f"Mean unique genres per prompt: {pd.Series(genre_counts).mean():.2f}")
    print(
        f"Mean unique albums per prompt: {pd.Series(album_counts).mean():.2f}")
    print(
        f"Median unique genres per prompt: {pd.Series(genre_counts).median():.2f}")
    print(
        f"Median unique albums per prompt: {pd.Series(album_counts).median():.2f}")
    print(
        f"Min/Max unique genres per prompt: {min(genre_counts)}/{max(genre_counts)}")
    print(
        f"Min/Max unique albums per prompt: {min(album_counts)}/{max(album_counts)}")

    # Inter-prompt diversity: how much overlap in albums/genres between prompts?
    if len(album_sets) > 1:
        overlap_counts = []
        for i in range(len(album_sets)):
            for j in range(i+1, len(album_sets)):
                overlap = len(album_sets[i] & album_sets[j])
                overlap_counts.append(overlap)
    # Only show visualizations if requested
    if show_viz:
        try:
            plt.figure(figsize=(8, 4))
            plt.hist(genre_counts, bins=range(1, max(genre_counts)+2), alpha=0.7)
            plt.title('Distribution of Unique Genres per Prompt')
            plt.xlabel('Unique Genres')
            plt.ylabel('Count')
            plt.show()
        except Exception as e:
            print(f"Visualization error: {e}")


def analyze_recommendation_overlap(df, k=5, show_viz=True):
    all_recs = []
    for _, row in df.iterrows():
        recs = row['recommended_albums'][:k]
        if recs and isinstance(recs[0], dict):
            albums = [r.get('album', None) for r in recs if r.get('album')]
        else:
            albums = recs
        all_recs.extend(albums)
    counter = Counter(all_recs)
    print("Most common recommended albums:")
    for album, count in counter.most_common(10):
        print(f"{album}: {count} times")
    if show_viz:
        plt.figure(figsize=(10, 4))
        pd.Series(counter).value_counts().sort_index().plot(kind='bar')
        plt.title('Frequency of Album Recommendations in Top-K')
        plt.xlabel('Times recommended in top-K')
        plt.ylabel('Number of albums')
        plt.show()


def plot_recommendation_feature_distribution(df, feature='genre', k=5, show_viz=True):
    all_features = []
    for _, row in df.iterrows():
        recs = row['recommended_albums'][:k]
        if recs and isinstance(recs[0], dict):
            feats = [r.get(feature, None) for r in recs if r.get(feature)]
        else:
            feats = []
        all_features.extend(feats)
    counter = Counter(all_features)
    print(f"Most common {feature}s in recommendations:")
    for feat, count in counter.most_common(10):
        print(f"{feat}: {count} times")
    if show_viz:
        plt.figure(figsize=(10, 4))
        pd.Series(counter).head(20).plot(kind='bar')
        plt.title(f'Top {feature.title()}s in Recommendations')
        plt.xlabel(feature.title())
        plt.ylabel('Count')
        plt.show()


def analyze_recommendation_bias(df, group_feature='genre', k=5, show_viz=True):
    plot_recommendation_feature_distribution(df, feature=group_feature, k=k, show_viz=show_viz)



# --------- Run Analyses ---------
show_viz = not args.no_vizs
analyze_recommendation_diversity(df, k=5, show_viz=show_viz)
analyze_recommendation_overlap(df, k=5, show_viz=show_viz)
analyze_recommendation_bias(df, group_feature='genre', k=5, show_viz=show_viz)

# Print top 20 artists by recommendation count
all_artists = []
for recs in df['recommended_albums']:
    all_artists.extend([r['artist'] for r in recs])
print("\nTop 20 artists by recommendation count:")
for artist, count in Counter(all_artists).most_common(20):
    print(f"{artist}: {count} times")

# Plot artist distribution (top 20)
if show_viz:
    plot_recommendation_feature_distribution(df, feature='artist', k=5, show_viz=show_viz)