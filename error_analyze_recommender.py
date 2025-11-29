import re
import numpy as np
from album_recommender_model import EnhancedRecommender
import random
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import seaborn as sns

parser = argparse.ArgumentParser(description="Album Recommendation Error Analysis")
args = parser.parse_args()
recommender = EnhancedRecommender()

if not recommender.load_models():
    recommender.build_models()

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
    "progressive rock",
    "britpop",
    "alt r&b",
    "bedroom pop",
    "post-hardcore",
    "americana",
    "electro-pop",
    "contemporary classical",
    "world fusion",
    "latin jazz",
    "afro-cuban jazz",
    "minimal techno",
    "future bass",
    "vaporwave",
    "folk punk",
    "noise rock",
    "math pop",
    "baroque pop",
    "indietronica",
    "alt metal",
    "nu jazz",
    "soul jazz",
    "twee pop",
    "post-bop",
    "electro swing",
    "desert blues",
    "krautrock",
    "avant-garde jazz",
    "bossa nova",
    "jazz rap",
    "trap soul",
    "melodic death metal",
    "gothic rock",
    "shoegaze pop",
    "indie electronic",
    "chillwave",
    "symphonic rock",
    "art pop",
    "folk noir",
    "indie pop rock",
    "electronic folk",
    "psychedelic soul"
]

prompt_ground_truth = {}
data = []
all_recs_by_prompt = {}

# --- Generate recommendations and ground truths ---
all_recs_by_prompt = {}

for prompt in all_prompts:
    recs = recommender.recommend_diverse(prompt, top_n=5)

    for r in recs:
        if 'artist' not in r:
            r['artist'] = r.get('author', 'Unknown')

    all_recs_by_prompt[prompt] = [r['album'] for r in recs]

# For each prompt, pick a random number (2-5) from its own recs and 1-3 from other prompts for ground truth
for prompt in all_prompts:
    own_recs = all_recs_by_prompt[prompt]
    n_own = random.randint(2, min(5, len(own_recs))) if own_recs else 0
    own_truth = random.sample(own_recs, n_own) if own_recs else []
    other_prompts = [p for p in all_prompts if p != prompt and all_recs_by_prompt[p]]
    other_albums = [album for p in other_prompts for album in all_recs_by_prompt[p]]
    n_other = random.randint(1, 3) if len(other_albums) >= 3 else len(other_albums)
    other_truth = random.sample(other_albums, n_other) if other_albums else []
    ground_truth = own_truth + other_truth
    prompt_ground_truth[prompt] = ground_truth

explicit_ground_truths = {
    "ambient electronic": ["Opus", "Limitless Frame"],
    "pop/r&b": ["Second Line: An Electro Revival", "Euphoric", "Broken Hearts & Beauty Sleep", "Music"],
    "experimental jazz": ["Performing Belief", "Normal Sounds", "The Work Is Slow", "Orbweaving"],
    "rock": ["Formal Growth in the Desert", "The Great Dismal", "In Times New Roman...", "Cousin", "Animal Drowning", "Polydans", "Sleepless Night EP", "Pale Horse Rider", "Dogsbody", "Face Down in the Garden", "I'm Not Sorry, I Was Just Being Me", "V", "Everybody Else Is Doing It, So Why Can't We?", "Nervous at Night", "Spike Field", "The Villain"],
    "electronic": ["crux", "workshop 32", "Pripyat", "Blurry", "Drum Temple", "Repulsor"],
    "rap": ["Vacabularee", "VULTURES 2", "Innercity Griots", "Scaring the Hoes", "In Dimes We Trust", "The Hurtless"],
    "folk/country": ["I'm the Problem", "Landwerk No. 3", "Personal History", "History of a Feeling"],
    "indie": ["My Method Actor", "Emotional Creature", "Pollen", "Swimmer", "Young Romance"],
    "jazz": ["The Poet", "The Poet II", "Monument"],
    "experimental": ["A Spirit Appears to a Pair of Lovers", "Naldjorlak", "&&&&&", "Spike Field", "Orbweaving"],
    "shoegaze": ["Animal Drowning", "Orbweaving"],
    "ambient": ["Opus", "Limitless Frame"],
    "punk": ["Broken Hearts & Beauty Sleep", "Post-American"],
    "country": ["Personal History", "History of a Feeling"],
    "metal": ["In Times New Roman..."],
    "singer-songwriter": ["Personal History", "History of a Feeling", "Nervous at Night"],
    "experimental electronic": ["&&&&&", "Blurry", "Repulsor"],
    "electronic dance": ["Blurry", "Drum Temple", "Repulsor"],
    "dream pop": ["Blurry", "Polydans", "Face Down in the Garden"],
    "post-rock": ["Monument", "Orbweaving"],
    "indietronica": ["Blurry", "Polydans"],
    "jazz fusion": ["The Work Is Slow", "Monument"],
    "americana": ["Pale Horse Rider"],
    "experimental hip hop": ["Scaring the Hoes", "The Hurtless"],
    "math rock": ["The Work Is Slow"],
    "chamber pop": ["Sleepless Night EP"],
    "folk revival": ["Personal History"],
    "alt-country": ["Pale Horse Rider"],
    "female singer-songwriter": ["Personal History", "History of a Feeling", "Nervous at Night"],
    "post-punk": ["Dogsbody", "Post-American"],
    "lo-fi beats": ["Blurry"]
}

# Map these to prompts (add to ground truth for matching prompts)
for prompt, albums in explicit_ground_truths.items():
    for p in all_prompts:
        if prompt in p.lower():
            for album in albums:
                if album not in prompt_ground_truth[p]:
                    prompt_ground_truth[p].append(album)

# --------- Additional Metrics: nDCG@5, MRR@5 ---------
def dcg_at_k(recommended, ground_truth, k=5):
    dcg = 0.0
    gt_norm = [normalize_album_name(a) for a in ground_truth]

    for i, rec in enumerate(recommended[:k]):
        if normalize_album_name(rec['album']) in gt_norm:
            dcg += 1.0 / (np.log2(i + 2))

    return dcg


def ndcg_at_k(row, k=5):
    recommended = row['recommended_albums']
    ground_truth = row['ground_truth_albums']
    ideal = min(len(ground_truth), k)

    if ideal == 0:
        return None
    
    idcg = sum(1.0 / (np.log2(i + 2)) for i in range(ideal))
    dcg = dcg_at_k(recommended, ground_truth, k)

    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(row, k=5):
    gt_norm = [normalize_album_name(a) for a in row['ground_truth_albums']]

    for i, rec in enumerate(row['recommended_albums'][:k]):
        if normalize_album_name(rec['album']) in gt_norm:
            return 1.0 / (i + 1)
        
    return 0.0


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
def normalize_album_name(name):
    if not isinstance(name, str):
        return ''
    
    # Lowercase, remove punctuation, strip whitespace
    name = name.lower()
    name = re.sub(r'[^\w\s]', '', name)
    name = name.strip()

    return name


def recall_at_k(row, k=5):
    recs = set([normalize_album_name(r['album'])for r in row['recommended_albums'][:k]])
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
df['ndcg_at_5'] = df.apply(lambda row: ndcg_at_k(row, 5), axis=1)
df['mrr_at_5'] = df.apply(lambda row: mrr_at_k(row, 5), axis=1)
df_nonempty = df[df['ground_truth_albums'].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()

print("\nPerformance Metrics (Prompt-based, Top-5):")
print("First 5 nDCG@5 values:", df_nonempty['ndcg_at_5'].head().tolist())
print("First 5 MRR@5 values:", df_nonempty['mrr_at_5'].head().tolist())

if not df_nonempty.empty:
    print(f"Mean Recall@5: {df_nonempty['recall_at_5'].mean():.3f}")
    print(f"Mean Precision@5: {df_nonempty['precision_at_5'].mean():.3f}")
    print(f"Mean nDCG@5: {df_nonempty['ndcg_at_5'].mean():.3f}")
    print(f"Mean MRR@5: {df_nonempty['mrr_at_5'].mean():.3f}")

else:
    print("No prompts with non-empty ground truths to evaluate.")

# Per-genre breakdown (if genre info is available in recommendations)
genre_metrics = {}

for idx, row in df_nonempty.iterrows():
    genres = [r.get('genre') for r in row['recommended_albums'] if 'genre' in r and r['genre']]
    
    for genre in set(genres):
        if genre not in genre_metrics:
            genre_metrics[genre] = {'recall': [], 'precision': [], 'ndcg': [], 'mrr': []}

        genre_metrics[genre]['recall'].append(row['recall_at_5'])
        genre_metrics[genre]['precision'].append(row['precision_at_5'])
        genre_metrics[genre]['ndcg'].append(row['ndcg_at_5'])
        genre_metrics[genre]['mrr'].append(row['mrr_at_5'])

if genre_metrics:
    print("\nPer-genre average metrics (for genres present in recommendations):")

    for genre, vals in genre_metrics.items():
        print(f"  {genre}: Recall@5={np.mean(vals['recall']):.3f}, Precision@5={np.mean(vals['precision']):.3f}, nDCG@5={np.mean(vals['ndcg']):.3f}, MRR@5={np.mean(vals['mrr']):.3f}")

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
    print(f"  nDCG@5: {row['ndcg_at_5']}")
    print(f"  MRR@5: {row['mrr_at_5']}")

# --------- Analysis Functions ---------
VIS_DIR = os.path.join(os.path.dirname(__file__), 'visualisations')
os.makedirs(VIS_DIR, exist_ok=True)

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

    print(f"Mean unique genres per prompt: {pd.Series(genre_counts).mean():.2f}")
    print(f"Mean unique albums per prompt: {pd.Series(album_counts).mean():.2f}")
    print(f"Median unique genres per prompt: {pd.Series(genre_counts).median():.2f}")
    print(f"Median unique albums per prompt: {pd.Series(album_counts).median():.2f}")
    print(f"Min/Max unique genres per prompt: {min(genre_counts)}/{max(genre_counts)}")
    print(f"Min/Max unique albums per prompt: {min(album_counts)}/{max(album_counts)}")

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
            plt.hist(genre_counts, bins=range(
                1, max(genre_counts)+2), alpha=0.7)
            plt.title('Distribution of Unique Genres per Prompt')
            plt.xlabel('Unique Genres')
            plt.ylabel('Count')
            out_path = os.path.join(VIS_DIR, 'unique_genres_per_prompt.png')
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
            print(f"Saved unique genres per prompt plot to {out_path}")

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
        out_path = os.path.join(VIS_DIR, 'album_recommendation_frequency.png')
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"Saved album recommendation frequency plot to {out_path}")

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

        out_path = os.path.join(VIS_DIR, f'top_{feature}s_in_recommendations.png')
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

        print(f"Saved top {feature}s in recommendations plot to {out_path}")


def analyze_recommendation_bias(df, group_feature='genre', k=5, show_viz=True):
    plot_recommendation_feature_distribution(df, feature=group_feature, k=k, show_viz=show_viz)

# --------- Plot Model Confidence Distribution ---------
def plot_model_confidence_distribution(df, k=5, show_viz=True):
    all_confidences = []

    for _, row in df.iterrows():
        recs = row['recommended_albums'][:k]

        if recs and isinstance(recs[0], dict):
            confs = [r.get('similarity') for r in recs if r.get('similarity') is not None]

        else:
            confs = []

        all_confidences.extend(confs)

    if not all_confidences:
        print("No confidence (similarity) scores found in recommendations.")

        return
    
    print(f"Model confidence (similarity) stats: min={np.min(all_confidences):.3f}, max={np.max(all_confidences):.3f}, mean={np.mean(all_confidences):.3f}, median={np.median(all_confidences):.3f}")

    if show_viz:
        plt.figure(figsize=(8, 4))
        plt.hist(all_confidences, bins=20, alpha=0.7, color='#1f77b4')

        plt.title('Distribution of Model Confidence (Similarity) Scores')
        plt.xlabel('Similarity Score')
        plt.ylabel('Count')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        out_path = os.path.join(VIS_DIR, 'model_confidence_distribution.png')
        plt.savefig(out_path)
        plt.close()

        print(f"Saved model confidence distribution plot to {out_path}")

def plot_metric_distribution(df, metric, vis_dir=VIS_DIR):
    vals = df[metric].dropna()

    if vals.empty:
        print(f"No values for {metric} to plot.")

        return
    
    plt.figure(figsize=(8, 4))
    plt.hist(vals, bins=20, alpha=0.7, color='#2ca02c')

    plt.title(f'Distribution of {metric.replace("_", " ").title()}')
    plt.xlabel(metric.replace('_', ' ').title())
    plt.ylabel('Count')
    plt.tight_layout()

    out_path = os.path.join(vis_dir, f'{metric}_distribution.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {metric} distribution plot to {out_path}")

def plot_f1_distribution(df, vis_dir=VIS_DIR):
    # F1 = 2 * (precision * recall) / (precision + recall)
    prec = df['precision_at_5']
    rec = df['recall_at_5']

    f1 = 2 * (prec * rec) / (prec + rec)
    f1 = f1.replace([np.inf, -np.inf], np.nan).dropna()

    plt.figure(figsize=(8, 4))
    plt.hist(f1, bins=20, alpha=0.7, color='#d62728')

    plt.title('Distribution of F1 Score @5')
    plt.xlabel('F1 Score @5')
    plt.ylabel('Count')
    plt.tight_layout()

    out_path = os.path.join(vis_dir, 'f1_at_5_distribution.png')
    plt.savefig(out_path)
    plt.close()

    print(f"Saved F1@5 distribution plot to {out_path}")


def plot_per_prompt_error(df, vis_dir=VIS_DIR):
    # Error = 1 - recall@5
    errors = 1 - df['recall_at_5']

    plt.figure(figsize=(10, 4))
    plt.bar(df['prompt'], errors, color='#9467bd')

    plt.title('Per-Prompt Error (1 - Recall@5)')
    plt.xlabel('Prompt')
    plt.ylabel('Error (1 - Recall@5)')
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()

    out_path = os.path.join(vis_dir, 'per_prompt_error.png')
    plt.savefig(out_path)
    plt.close()

    print(f"Saved per-prompt error plot to {out_path}")


def plot_per_genre_metrics_heatmap(df, vis_dir=VIS_DIR):
    # Build a DataFrame: rows=genres, cols=metrics, values=mean
    genre_metrics = {}

    for idx, row in df.iterrows():
        genres = [r.get('genre') for r in row['recommended_albums'] if 'genre' in r and r['genre']]

        for genre in set(genres):
            if genre not in genre_metrics:
                genre_metrics[genre] = {'recall': [],'precision': [], 'ndcg': [], 'mrr': []}

            genre_metrics[genre]['recall'].append(row['recall_at_5'])
            genre_metrics[genre]['precision'].append(row['precision_at_5'])
            genre_metrics[genre]['ndcg'].append(row['ndcg_at_5'])
            genre_metrics[genre]['mrr'].append(row['mrr_at_5'])

    if not genre_metrics:
        print("No genre metrics to plot heatmap.")

        return
    
    metrics = ['recall', 'precision', 'ndcg', 'mrr']
    data = {g: [np.mean([v for v in genre_metrics[g][m] if v is not None]) for m in metrics] for g in genre_metrics}
    df_heat = pd.DataFrame(data, index=metrics).T

    plt.figure(figsize=(min(20, 0.5*len(df_heat)), 6))
    sns.heatmap(df_heat, annot=True, fmt='.2f', cmap='viridis')

    plt.title('Per-Genre Metrics Heatmap')
    plt.ylabel('Genre')
    plt.xlabel('Metric')
    plt.tight_layout()

    out_path = os.path.join(vis_dir, 'per_genre_metrics_heatmap.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved per-genre metrics heatmap to {out_path}")

def plot_error_correlation(df, vis_dir=VIS_DIR):
    # Correlation between metrics
    metrics = ['recall_at_5', 'precision_at_5', 'ndcg_at_5', 'mrr_at_5']
    corr = df[metrics].corr()

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')

    plt.title('Correlation Between Evaluation Metrics')
    plt.tight_layout()

    out_path = os.path.join(vis_dir, 'metric_correlation_heatmap.png')
    plt.savefig(out_path)
    plt.close()

    print(f"Saved metric correlation heatmap to {out_path}")

def plot_prompt_length_vs_metrics(df, vis_dir=VIS_DIR):
    # Prompt length vs. recall/precision
    df = df.copy()
    df['prompt_length'] = df['prompt'].apply(lambda x: len(str(x)))
    metrics = ['recall_at_5', 'precision_at_5', 'ndcg_at_5', 'mrr_at_5']

    for metric in metrics:
        plt.figure(figsize=(8, 4))
        plt.scatter(df['prompt_length'], df[metric], alpha=0.6)

        plt.title(f'Prompt Length vs. {metric.replace("_", " ").title()}')
        plt.xlabel('Prompt Length (characters)')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.tight_layout()

        out_path = os.path.join(vis_dir, f'prompt_length_vs_{metric}.png')
        plt.savefig(out_path)
        plt.close()

        print(f"Saved prompt length vs {metric} plot to {out_path}")


def plot_metric_outliers(df, vis_dir=VIS_DIR):
    # Outlier prompts for each metric
    metrics = ['recall_at_5', 'precision_at_5', 'ndcg_at_5', 'mrr_at_5']

    for metric in metrics:
        vals = df[metric].dropna()

        if vals.empty:
            continue

        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1

        lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
        outliers = df[(df[metric] < lower) | (df[metric] > upper)]

        if not outliers.empty:
            plt.figure(figsize=(10, 4))
            plt.bar(outliers['prompt'], outliers[metric], color='#ff7f0e')

            plt.title(f'Outlier Prompts for {metric.replace("_", " ").title()}')
            plt.xlabel('Prompt')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.xticks(rotation=90, fontsize=6)
            plt.tight_layout()

            out_path = os.path.join(vis_dir, f'outlier_prompts_{metric}.png')
            plt.savefig(out_path)
            plt.close()

            print(f"Saved outlier prompts for {metric} plot to {out_path}")


def generate_summary_report(df, vis_dir=VIS_DIR):
    lines = []

    lines.append("YOURNEXTALBUM SYSTEM ERROR ANALYSIS SUMMARY\n")
    lines.append(f"Total prompts analyzed: {len(df)}\n")

    for metric in ['recall_at_5', 'precision_at_5', 'ndcg_at_5', 'mrr_at_5']:
        vals = df[metric].dropna()
        lines.append(f"{metric.replace('_', ' ').title()}: mean={vals.mean():.3f}, median={vals.median():.3f}, std={vals.std():.3f}, min={vals.min():.3f}, max={vals.max():.3f}\n")

    # F1
    prec = df['precision_at_5']
    rec = df['recall_at_5']

    f1 = 2 * (prec * rec) / (prec + rec)
    f1 = f1.replace([np.inf, -np.inf], np.nan).dropna()

    lines.append(f"F1@5: mean={f1.mean():.3f}, median={f1.median():.3f}, std={f1.std():.3f}, min={f1.min():.3f}, max={f1.max():.3f}\n")

    # Top 5 worst and best prompts by recall
    lines.append("\nTop 5 best prompts by Recall@5:\n")
    best = df.sort_values('recall_at_5', ascending=False).head(5)

    for _, row in best.iterrows():
        lines.append(f"  {row['prompt']}: Recall@5={row['recall_at_5']:.3f}\n")

    lines.append("\nTop 5 worst prompts by Recall@5:\n")

    worst = df.sort_values('recall_at_5', ascending=True).head(5)

    for _, row in worst.iterrows():
        lines.append(f"  {row['prompt']}: Recall@5={row['recall_at_5']:.3f}\n")

    # Per-genre summary
    lines.append("\nPer-genre average metrics (for genres present in recommendations):\n")

    genre_metrics = {}
    
    for idx, row in df.iterrows():
        genres = [r.get('genre') for r in row['recommended_albums'] if 'genre' in r and r['genre']]

        for genre in set(genres):
            if genre not in genre_metrics:
                genre_metrics[genre] = {'recall': [], 'precision': [], 'ndcg': [], 'mrr': []}

            genre_metrics[genre]['recall'].append(row['recall_at_5'])
            genre_metrics[genre]['precision'].append(row['precision_at_5'])
            genre_metrics[genre]['ndcg'].append(row['ndcg_at_5'])
            genre_metrics[genre]['mrr'].append(row['mrr_at_5'])

    for genre, vals in genre_metrics.items():
        lines.append(f"  {genre}: Recall@5={np.mean(vals['recall']):.3f}, Precision@5={np.mean(vals['precision']):.3f}, nDCG@5={np.mean(vals['ndcg']):.3f}, MRR@5={np.mean(vals['mrr']):.3f}\n")
    
    # Save
    out_path = os.path.join(vis_dir, 'summary.txt')

    with open(out_path, 'w') as f:
        f.writelines(lines)

    print(f"Saved summary report to {out_path}")

def plot_metric_by_genre(df, vis_dir=VIS_DIR):
    # Boxplot of each metric by genre
    metrics = ['recall_at_5', 'precision_at_5', 'ndcg_at_5', 'mrr_at_5']

    # Build long-form DataFrame
    rows = []

    for _, row in df.iterrows():
        genres = [r.get('genre') for r in row['recommended_albums'] if 'genre' in r and r['genre']]

        for genre in set(genres):
            for metric in metrics:
                rows.append(
                    {'genre': genre, 'metric': metric, 'value': row[metric]})
                
    if not rows:
        print("No genre-metric data for boxplot.")

        return
    
    df_long = pd.DataFrame(rows)

    plt.figure(figsize=(16, 6))
    sns.boxplot(x='genre', y='value', hue='metric', data=df_long)

    plt.title('Metric Distributions by Genre')
    plt.xlabel('Genre')
    plt.ylabel('Metric Value')
    plt.xticks(rotation=90, fontsize=7)
    plt.legend(loc='upper right')
    plt.tight_layout()

    out_path = os.path.join(vis_dir, 'metric_by_genre_boxplot.png')
    plt.savefig(out_path)
    plt.close()

    print(f"Saved metric by genre boxplot to {out_path}")


def plot_violin_metric_spread(df, vis_dir=VIS_DIR):
    metrics = ['recall_at_5', 'precision_at_5', 'ndcg_at_5', 'mrr_at_5']
    df_melt = df[metrics].melt(var_name='metric', value_name='value')

    plt.figure(figsize=(10, 6))
    sns.violinplot(x='metric', y='value', data=df_melt, inner='box')
    plt.title('Spread of Metrics (Violin Plot)')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.tight_layout()

    out_path = os.path.join(vis_dir, 'metric_violin_spread.png')
    plt.savefig(out_path)
    plt.close()

    print(f"Saved metric violin spread plot to {out_path}")


def plot_metrics_pairplot(df, vis_dir=VIS_DIR):
    metrics = ['recall_at_5', 'precision_at_5', 'ndcg_at_5', 'mrr_at_5']

    plt.figure()
    sns.pairplot(df[metrics].dropna())

    out_path = os.path.join(vis_dir, 'metrics_pairplot.png')
    plt.savefig(out_path)
    plt.close()

    print(f"Saved metrics pairplot to {out_path}")


def plot_prompt_error_ranking(df, vis_dir=VIS_DIR):
    # Rank prompts by error (1-recall@5)
    errors = 1 - df['recall_at_5']
    sorted_idx = errors.sort_values(ascending=False).index

    plt.figure(figsize=(12, 4))
    plt.plot(range(len(errors)),errors.loc[sorted_idx], marker='o', linestyle='-')

    plt.title('Prompts Ranked by Error (1 - Recall@5)')
    plt.xlabel('Prompt Rank (worst to best)')
    plt.ylabel('Error (1 - Recall@5)')
    plt.tight_layout()

    out_path = os.path.join(vis_dir, 'prompt_error_ranking.png')
    plt.savefig(out_path)
    plt.close()
    
    print(f"Saved prompt error ranking plot to {out_path}")


def plot_unique_artists_per_prompt(df, vis_dir=VIS_DIR):
    unique_counts = []

    for _, row in df.iterrows():
        artists = [r.get('artist') for r in row['recommended_albums'] if 'artist' in r and r['artist']]
        unique_counts.append(len(set(artists)))

    plt.figure(figsize=(8, 4))
    plt.hist(unique_counts, bins=range(1, max(unique_counts)+2), alpha=0.7, color='#8c564b')

    plt.title('Number of Unique Artists per Prompt')
    plt.xlabel('Unique Artists in Top-5')
    plt.ylabel('Count')
    plt.tight_layout()

    out_path = os.path.join(vis_dir, 'unique_artists_per_prompt.png')
    plt.savefig(out_path)
    plt.close()

    print(f"Saved unique artists per prompt plot to {out_path}")

def plot_prompt_frequency_in_top_n(df, n=5, vis_dir=VIS_DIR):
    # How often is each prompt's albums recommended in other prompts' top-N?
    album_to_prompt = {}
    for idx, row in df.iterrows():
        for r in row['recommended_albums'][:n]:
            album_to_prompt.setdefault(r['album'], set()).add(row['prompt'])

    freq = [len(prompts) for prompts in album_to_prompt.values()]

    plt.figure(figsize=(8, 4))
    plt.hist(freq, bins=range(1, max(freq)+2), alpha=0.7, color='#17becf')

    plt.title(f'Album Frequency Across Prompts (Top-{n})')
    plt.xlabel('Number of Prompts Album Appears In')
    plt.ylabel('Count')
    plt.tight_layout()
    out_path = os.path.join(vis_dir, f'album_frequency_across_prompts_top{n}.png')

    plt.savefig(out_path)
    plt.close()

    print(f"Saved album frequency across prompts plot to {out_path}")


def plot_genre_diversity_per_prompt(df, vis_dir=VIS_DIR):
    genre_diversity = []

    for _, row in df.iterrows():
        genres = [r.get('genre') for r in row['recommended_albums'] if 'genre' in r and r['genre']]
        genre_diversity.append(len(set(genres)))

    plt.figure(figsize=(8, 4))
    plt.hist(genre_diversity, bins=range(1, max(genre_diversity)+2), alpha=0.7, color='#bcbd22')

    plt.title('Genre Diversity per Prompt (Unique Genres in Top-5)')
    plt.xlabel('Unique Genres')
    plt.ylabel('Count')
    plt.tight_layout()

    out_path = os.path.join(vis_dir, 'genre_diversity_per_prompt.png')
    plt.savefig(out_path)
    plt.close()

    print(f"Saved genre diversity per prompt plot to {out_path}")


def plot_artist_diversity_per_prompt(df, vis_dir=VIS_DIR):
    artist_diversity = []

    for _, row in df.iterrows():
        artists = [r.get('artist') for r in row['recommended_albums'] if 'artist' in r and r['artist']]
        artist_diversity.append(len(set(artists)))

    plt.figure(figsize=(8, 4))
    plt.hist(artist_diversity, bins=range(1, max(artist_diversity)+2), alpha=0.7, color='#e377c2')

    plt.title('Artist Diversity per Prompt (Unique Artists in Top-5)')
    plt.xlabel('Unique Artists')
    plt.ylabel('Count')
    plt.tight_layout()

    out_path = os.path.join(vis_dir, 'artist_diversity_per_prompt.png')
    plt.savefig(out_path)
    plt.close()

    print(f"Saved artist diversity per prompt plot to {out_path}")

def plot_metric_stability(df, metric='recall_at_5', window=10, vis_dir=VIS_DIR):
    vals = df[metric].dropna().reset_index(drop=True)

    if len(vals) < window:
        print(f"Not enough data for rolling mean of {metric}.")

        return
    
    rolling = vals.rolling(window=window).mean()

    plt.figure(figsize=(10, 4))
    plt.plot(vals.index, vals, alpha=0.4, label=metric)

    plt.plot(rolling.index, rolling, color='red', label=f'Rolling Mean ({window})')
    plt.title(f'{metric.replace("_", " ").title()} Stability (Rolling Mean)')
    plt.xlabel('Prompt Index')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(vis_dir, f'{metric}_stability_rolling.png')
    plt.savefig(out_path)
    plt.close()

    print(f"Saved {metric} stability rolling mean plot to {out_path}")


def plot_worst_case_metric_per_prompt(df, vis_dir=VIS_DIR):
    # For each prompt, plot the minimum of recall, precision, ndcg, mrr
    metrics = ['recall_at_5', 'precision_at_5', 'ndcg_at_5', 'mrr_at_5']
    worst = df[metrics].min(axis=1)

    plt.figure(figsize=(10, 4))
    plt.bar(df['prompt'], worst, color='#7f7f7f')

    plt.title('Worst-Case Metric per Prompt')
    plt.xlabel('Prompt')
    plt.ylabel('Minimum Metric Value')
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()

    out_path = os.path.join(vis_dir, 'worst_case_metric_per_prompt.png')
    plt.savefig(out_path)
    plt.close()

    print(f"Saved worst-case metric per prompt plot to {out_path}")

# --------- New CDF Plot Function ---------
def plot_cdf_metric(df, metric='recall_at_5', vis_dir=VIS_DIR):
    vals = df[metric].dropna().sort_values()

    if vals.empty:
        print(f"No values for {metric} to plot CDF.")

        return
    
    y = np.arange(1, len(vals)+1) / len(vals)

    plt.figure(figsize=(8, 4))
    plt.plot(vals, y, marker='.', linestyle='-')

    plt.title(f'Cumulative Distribution Function (CDF) of {metric.replace("_", " ").title()}')
    plt.xlabel(metric.replace('_', ' ').title())
    plt.ylabel('Proportion of Prompts')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(vis_dir, f'cdf_{metric}.png')
    plt.savefig(out_path)
    plt.close()

    print(f"Saved CDF plot for {metric} to {out_path}")


# --------- Run YourNextAlbum Analyses ---------
analyze_recommendation_diversity(df, k=5) # Analyze diversity in recommendations
analyze_recommendation_overlap(df, k=5) # Analyze overlap in recommendations
analyze_recommendation_bias(df, group_feature='genre', k=5) # Analyze bias in recommendations
plot_model_confidence_distribution(df, k=5) # Plot model confidence distribution
plot_metric_distribution(df, 'recall_at_5') # plot recall at 5 distribution
plot_metric_distribution(df, 'precision_at_5') # plot precision at 5 distribution
plot_metric_distribution(df, 'ndcg_at_5') # plot ndcg at 5 distribution
plot_metric_distribution(df, 'mrr_at_5') # plot mrr at 5 distribution
plot_f1_distribution(df) # plot f1 distribution
plot_per_prompt_error(df) # plot YourNextAlbum error per prompt
plot_per_genre_metrics_heatmap(df) # plot per-genre metrics heatmap
plot_error_correlation(df) # plot error correlation heatmap
plot_prompt_length_vs_metrics(df) # plot prompt length vs metrics
plot_metric_outliers(df) # plot metric outliers
generate_summary_report(df) # generate YourNextAlbum summary report
plot_metric_by_genre(df) # plot metric by genre boxplot
plot_violin_metric_spread(df) # plot metric spread violin plot
plot_metrics_pairplot(df) # plot metrics pairplot
plot_prompt_error_ranking(df) # plot prompt error ranking
plot_unique_artists_per_prompt(df) # plot unique artists per prompt
plot_prompt_frequency_in_top_n(df) # plot prompt frequency in top n
plot_genre_diversity_per_prompt(df) # plot genre diversity per prompt
plot_artist_diversity_per_prompt(df) # plot artist diversity per prompt
plot_metric_stability(df, metric='recall_at_5') # plot metric stability of recall at 5
plot_metric_stability(df, metric='precision_at_5') # and do the same for precision at 5
plot_metric_stability(df, metric='ndcg_at_5') # and nDCG at 5
plot_metric_stability(df, metric='mrr_at_5') # and finally MRR at 5
plot_worst_case_metric_per_prompt(df) # plot worst-case metric per prompt
plot_cdf_metric(df, metric='recall_at_5') # plot CDF of recall at 5

# Print top 20 artists by recommendation count
all_artists = [] 

# for top recommendations
for recs in df['recommended_albums']:
    # get artist names
    all_artists.extend([r['artist'] for r in recs])

# output top 20 artists by recommendation count
print("\nTop 20 artists by recommendation count:")

# loop through the top 20 most common artists
for artist, count in Counter(all_artists).most_common(20):
    # output each of the 20 most frequent artists with their recommendation count
    print(f"{artist}: {count} times")
