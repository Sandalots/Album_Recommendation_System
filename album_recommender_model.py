import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle


class EnhancedRecommender:
    def __init__(self, data_path='outputs/pitchfork_reviews_preprocessed_plus_sentiments.csv'):
        self.df = pd.read_csv(data_path)
        self.vectorizer = None
        self.tfidf_matrix = None
        self.embeddings = None
        self.model = None

        # Check if enhanced preprocessing features are available
        self.has_enhanced_features = all(col in self.df.columns for col in
                                         ['review_text_processed', 'unique_word_ratio', 'score_normalized'])
        if self.has_enhanced_features:
            pass  # Enhanced features detected silently

    def build_models(self):
        # Create enriched feature vectors with all available information
        feature_components = [
            self.df['genre'].astype(str),
            self.df['themes'].astype(str),
            self.df['artist_name'].astype(str),
            self.df['album_name'].astype(str),
            self.df['key_highlights'].astype(str)
        ]

        # Add lyrical themes if available
        if 'lyrical_themes' in self.df.columns:
            feature_components.append(
                self.df['lyrical_themes'].fillna('').astype(str))

        # Add musical characteristics if available
        if 'musical_characteristics' in self.df.columns:
            feature_components.append(
                self.df['musical_characteristics'].fillna('').astype(str))

        # Add NEW enhanced features
        if 'instrumentation' in self.df.columns:
            feature_components.append(
                self.df['instrumentation'].fillna('').astype(str))
        if 'mood_energy' in self.df.columns:
            feature_components.append(
                self.df['mood_energy'].fillna('').astype(str))
        if 'listening_contexts' in self.df.columns:
            feature_components.append(
                self.df['listening_contexts'].fillna('').astype(str))
        if 'production_quality' in self.df.columns:
            feature_components.append(
                self.df['production_quality'].fillna('').astype(str))

        # Combine all features with weighted repetition for emphasis
        self.df['combined_features'] = (
            feature_components[0] + ' ' +
            feature_components[1] + ' ' + feature_components[1] + ' '
        )

        # Add remaining components
        for comp in feature_components[2:]:
            self.df['combined_features'] = self.df['combined_features'] + ' ' + comp

        # If enhanced preprocessing is available, use processed text for better TF-IDF
        if self.has_enhanced_features and 'review_text_processed' in self.df.columns:
            # Using preprocessed album review text for TF-IDF (silent)
            # Add processed review text to features for better matching
            self.df['combined_features'] = (self.df['combined_features'] + ' ' +
                                            self.df['review_text_processed'].fillna(''))

        # Building TF-IDF model (silent)
        self.vectorizer = TfidfVectorizer(
            max_features=8000,  # Increased from 5000
            stop_words='english',
            ngram_range=(1, 3),  # Include trigrams for better phrase matching
            min_df=2,  # Ignore very rare terms
            max_df=0.8  # Ignore very common terms
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df['combined_features'])

        # Building semantic embeddings (silent)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Create richer text for embeddings including artist and album names
        embedding_texts = []
        for idx, row in self.df.iterrows():
            text_parts = [
                f"Artist: {row['artist_name']}",
                f"Album: {row['album_name']}",
                f"Genre: {row['genre']}",
                f"Themes: {row['themes']}",
                f"Review highlights: {row['key_highlights']}"
            ]
            # Use preprocessed text if available (more efficient for embeddings)
            if self.has_enhanced_features and 'review_text_processed' in self.df.columns:
                if pd.notna(row['review_text_processed']) and row['review_text_processed']:
                    text_parts.append(
                        f"Review: {row['review_text_processed'][:500]}")

            if 'lyrical_themes' in self.df.columns and pd.notna(row['lyrical_themes']) and row['lyrical_themes']:
                text_parts.append(f"Lyrical content: {row['lyrical_themes']}")
            if 'comparisons' in self.df.columns and pd.notna(row['comparisons']) and row['comparisons']:
                text_parts.append(f"Similar to: {row['comparisons']}")

            # Add NEW enhanced features to embeddings
            if 'instrumentation' in self.df.columns and pd.notna(row['instrumentation']) and row['instrumentation']:
                text_parts.append(f"Instruments: {row['instrumentation']}")
            if 'mood_energy' in self.df.columns and pd.notna(row['mood_energy']) and row['mood_energy']:
                text_parts.append(f"Mood and energy: {row['mood_energy']}")
            if 'listening_contexts' in self.df.columns and pd.notna(row['listening_contexts']) and row['listening_contexts']:
                text_parts.append(f"Best for: {row['listening_contexts']}")

            embedding_texts.append(' '.join(text_parts))

        self.embeddings = self.model.encode(
            embedding_texts,
            show_progress_bar=True,
            batch_size=32
        )

        print("Album recommendation models built.")

    def recommend_by_mood(self, mood_description, top_n=5, sentiment_filter=None, min_score=None):
        """
        Recommend albums based on mood/vibe description

        Args:
            mood_description: e.g., "upbeat and energetic" or "dark and moody"
            sentiment_filter: 'POSITIVE' or 'NEGATIVE' to filter by review sentiment
            min_score: minimum album score
        """
        if self.embeddings is None:
            raise ValueError("Models not built. Call build_models() first.")

        # Make query case-insensitive
        mood_description_lower = mood_description.lower()
        user_embedding = self.model.encode([mood_description_lower])
        similarities = cosine_similarity(
            user_embedding, self.embeddings).flatten()

        results_df = self.df.copy()
        results_df['similarity'] = similarities

        if sentiment_filter:
            results_df = results_df[results_df['sentiment_label']
                                    == sentiment_filter]

        if min_score is not None:
            results_df = results_df[results_df['score'] >= min_score]

        recommendations = results_df.nlargest(top_n, 'similarity')
        return self._format_recommendations(recommendations)

    def recommend_by_themes(self, desired_themes, top_n=5, min_score=7.0):
        """
        Recommend albums that match specific themes

        Args:
            desired_themes: list like ['experimental', 'atmospheric'] or single string
        """
        if isinstance(desired_themes, str):
            desired_themes = [desired_themes]

        def theme_match_score(themes_str):
            if pd.isna(themes_str):
                return 0
            album_themes = themes_str.lower().split(', ')
            return sum(1 for theme in desired_themes if theme.lower() in album_themes)

        results_df = self.df.copy()
        results_df['theme_match'] = results_df['themes'].apply(
            theme_match_score)

        results_df = results_df[results_df['theme_match'] > 0]

        if min_score is not None:
            results_df = results_df[results_df['score'] >= min_score]

        recommendations = results_df.nlargest(top_n, ['theme_match', 'score'])
        return self._format_recommendations(recommendations)

    def recommend_similar_albums(self, album_name, artist_name=None, top_n=5):
        """Find albums similar to a specific album"""
        if artist_name:
            target = self.df[
                (self.df['album_name'].str.contains(album_name, case=False, na=False)) &
                (self.df['artist_name'].str.contains(
                    artist_name, case=False, na=False))
            ]
        else:
            target = self.df[self.df['album_name'].str.contains(
                album_name, case=False, na=False)]

        if len(target) == 0:
            return []

        target_idx = target.index[0]
        target_embedding = self.embeddings[target_idx].reshape(1, -1)

        similarities = cosine_similarity(
            target_embedding, self.embeddings).flatten()

        results_df = self.df.copy()
        results_df['similarity'] = similarities
        results_df = results_df[results_df.index != target_idx]

        recommendations = results_df.nlargest(top_n, 'similarity')
        return self._format_recommendations(recommendations)

    def recommend_by_sentiment_and_genre(self, genre, sentiment='POSITIVE', top_n=5, min_score=7.0):
        """Find albums by genre with specific sentiment"""
        filtered = self.df[
            (self.df['genre'].str.contains(genre, case=False, na=False)) &
            (self.df['sentiment_label'] == sentiment) &
            (self.df['score'] >= min_score)
        ]

        recommendations = filtered.nlargest(top_n, 'sentiment_score')
        return self._format_recommendations(recommendations)

    def recommend_by_instruments(self, instruments, top_n=5, min_score=7.0):
        """Find albums featuring specific instruments

        Args:
            instruments: list of instruments like ['guitar', 'synthesizer'] or single string
        """
        if isinstance(instruments, str):
            instruments = [instruments]

        if 'instrumentation' not in self.df.columns:
            # Instrumentation data not available (silent)
            return []

        def instrument_match_score(instr_str):
            if pd.isna(instr_str):
                return 0
            album_instruments = instr_str.lower().split(', ')
            return sum(1 for instr in instruments if instr.lower() in album_instruments)

        results_df = self.df.copy()
        results_df['instrument_match'] = results_df['instrumentation'].apply(
            instrument_match_score)
        results_df = results_df[results_df['instrument_match'] > 0]

        if min_score is not None:
            results_df = results_df[results_df['score'] >= min_score]

        recommendations = results_df.nlargest(
            top_n, ['instrument_match', 'score'])
        return self._format_recommendations(recommendations)

    def recommend_by_mood_energy(self, mood=None, energy_level=None, top_n=5, min_score=7.0):
        """Find albums by mood and/or energy level

        Args:
            mood: mood keyword like 'sad', 'happy', 'dark', 'calm'
            energy_level: 'high', 'medium', or 'low'
        """
        if 'mood_energy' not in self.df.columns:
            # Mood/energy data not available (silent)
            return []

        results_df = self.df.copy()

        # Filter by mood
        if mood:
            results_df = results_df[results_df['mood_energy'].str.contains(
                mood, case=False, na=False)]

        # Filter by energy level
        if energy_level:
            results_df = results_df[results_df['mood_energy'].str.contains(
                f"energy:{energy_level}", case=False, na=False)]

        if min_score is not None:
            results_df = results_df[results_df['score'] >= min_score]

        recommendations = results_df.nlargest(top_n, 'score')
        return self._format_recommendations(recommendations)

    def recommend_by_listening_context(self, context, top_n=5, min_score=7.0):
        """Find albums suitable for specific listening contexts

        Args:
            context: 'party', 'study', 'workout', 'relaxation', 'driving', 'intimate', 'headphones'
        """
        if 'listening_contexts' not in self.df.columns:
            print("️ Listening context data not available")
            return []

        results_df = self.df[
            (self.df['listening_contexts'].str.contains(context, case=False, na=False)) &
            (self.df['score'] >= min_score)
        ]

        recommendations = results_df.nlargest(top_n, 'score')
        return self._format_recommendations(recommendations)

    def recommend_innovative_albums(self, top_n=5, min_score=7.5):
        """Find the most innovative/groundbreaking albums"""
        if 'novelty_score' not in self.df.columns:
            print("️ Novelty score data not available")
            return []

        results_df = self.df[
            (self.df['novelty_score'] > 0) &
            (self.df['score'] >= min_score)
        ]

        recommendations = results_df.nlargest(
            top_n, ['novelty_score', 'score'])
        return self._format_recommendations(recommendations)

    def recommend_by_artist_influence(self, artist_name, top_n=5, min_score=7.0):
        """Find albums influenced by or similar to a specific artist"""
        if 'comparisons' not in self.df.columns:
            print("️ Artist comparison data not available")
            return []

        results_df = self.df[
            (self.df['comparisons'].str.contains(artist_name, case=False, na=False)) &
            (self.df['score'] >= min_score)
        ]

        if len(results_df) == 0:
            # Fallback to semantic search
            return self.recommend_by_mood(f"music similar to {artist_name}", top_n, min_score=min_score)

        recommendations = results_df.nlargest(top_n, 'score')
        return self._format_recommendations(recommendations)

    def recommend_diverse(self, mood_description, top_n=5, diversity_weight=0.3, min_score=None):
        """
        Recommend albums with controlled diversity to avoid repetitive themes

        Args:
            mood_description: user's search query
            top_n: number of recommendations
            diversity_weight: 0-1, higher = more diverse results
            min_score: minimum album score filter
        """
        if self.embeddings is None:
            raise ValueError("Models not built. Call build_models() first.")

        # First check for exact/partial album or artist name matches
        query_lower = mood_description.lower()
        exact_matches = self.df[
            (self.df['album_name'].str.lower().str.contains(query_lower, na=False, regex=False)) |
            (self.df['artist_name'].str.lower().str.contains(
                query_lower, na=False, regex=False))
        ]

        # Get initial larger pool of candidates
        user_embedding = self.model.encode([mood_description])
        similarities = cosine_similarity(
            user_embedding, self.embeddings).flatten()

        results_df = self.df.copy()
        results_df['similarity'] = similarities

        # Boost similarity scores for exact name matches
        if not exact_matches.empty:
            for idx in exact_matches.index:
                results_df.loc[idx, 'similarity'] = min(
                    1.0, results_df.loc[idx, 'similarity'] + 0.5)

        if min_score is not None:
            results_df = results_df[results_df['score'] >= min_score]

        # Get top candidates (3x what we need)
        candidates = results_df.nlargest(top_n * 3, 'similarity')

        # Diversification algorithm
        selected = []
        selected_themes = set()

        for idx, row in candidates.iterrows():
            if len(selected) >= top_n:
                break

            album_themes = set(row['themes'].split(
                ', ')) if pd.notna(row['themes']) else set()

            # Calculate theme overlap with already selected albums
            theme_overlap = len(
                album_themes & selected_themes) / max(len(album_themes), 1)

            # Adjust similarity score based on diversity
            diversity_penalty = theme_overlap * diversity_weight
            adjusted_score = row['similarity'] * (1 - diversity_penalty)

            # Add to selection
            selected.append({
                'idx': idx,
                'adjusted_score': adjusted_score,
                'original_similarity': row['similarity']
            })
            selected_themes.update(album_themes)

        # Re-sort by adjusted scores and get final recommendations
        selected = sorted(
            selected, key=lambda x: x['adjusted_score'], reverse=True)
        final_indices = [s['idx'] for s in selected[:top_n]]

        recommendations = results_df.loc[final_indices].copy()
        recommendations['similarity'] = [s['adjusted_score']
                                         for s in selected[:top_n]]

        return self._format_recommendations(recommendations)

    def _format_recommendations(self, recommendations_df):
        results = []
        for idx, row in recommendations_df.iterrows():
            result = {
                'artist': row['artist_name'],
                'album': row['album_name'],
                'genre': row['genre'],
                'score': row['score'],
                'year': row['release_year'],
                'sentiment': row.get('sentiment_label', 'N/A'),
                'sentiment_confidence': row.get('sentiment_score', 0),
                'themes': row.get('themes', 'N/A'),
                'highlights': row.get('key_highlights', ''),
                'url': row['url'],
                'similarity': row.get('similarity', None),
                'theme_match': row.get('theme_match', None),
                # NEW FEATURES
                'instrumentation': row.get('instrumentation', ''),
                'mood_energy': row.get('mood_energy', ''),
                'listening_contexts': row.get('listening_contexts', ''),
                'is_polarizing': row.get('is_polarizing', False),
                'novelty_score': row.get('novelty_score', 0),
                'critical_consensus': row.get('critical_consensus', 'unknown')
            }
            results.append(result)
        return results

    def display_recommendations(self, recommendations):
        print("\n" + "="*80)
        print("ALBUM RECOMMENDATIONS (TF-IDF, Sentiment, Semantic)")
        print("="*80 + "\n")

        if len(recommendations) == 0:
            print("No recommendations found matching your criteria.\n")
            return

        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['artist']} - {rec['album']}")
            print(
                f"   Genre: {rec['genre']} | Score: {rec['score']} | Year: {rec['year']}")
            print(
                f"   Sentiment: {rec['sentiment']} (confidence: {rec['sentiment_confidence']:.2f})")
            print(f"   Themes: {rec['themes']}")

            # Show new enhanced features
            if rec.get('instrumentation'):
                print(f"   Instrumentation: {rec['instrumentation']}")
            if rec.get('mood_energy'):
                print(f"   Mood/Energy: {rec['mood_energy']}")
            if rec.get('listening_contexts'):
                print(f"   Best for: {rec['listening_contexts']}")
            if rec.get('is_polarizing'):
                print(f"   ⚠️  Polarizing/Divisive album")
            if rec.get('novelty_score', 0) > 0:
                print(f"   💡 Innovative (novelty: +{rec['novelty_score']})")
            elif rec.get('novelty_score', 0) < 0:
                print(f"   📋 Derivative (novelty: {rec['novelty_score']})")

            if rec['similarity']:
                print(f"   Match Score: {rec['similarity']:.3f}")
            if rec['theme_match']:
                print(f"   Theme Matches: {rec['theme_match']}")

            highlights = rec['highlights'].split(' | ')
            if highlights and highlights[0]:
                print(f"   Key Points:")
                for j, highlight in enumerate(highlights[:2], 1):
                    print(f"     • {highlight[:150]}...")

            print(f"   Full Review: {rec['url']}")
            print()

    def load_models(self, directory='models'):
        """Load pre-built models from disk if available"""
        vectorizer_path = f'{directory}/tfidf_vectorizer.pkl'
        tfidf_path = f'{directory}/album_tfidf_matrix.pkl'
        embeddings_path = f'{directory}/album_semantic_embeddings.npy'

        if not all(os.path.exists(p) for p in [vectorizer_path, tfidf_path, embeddings_path]):
            return False

        try:
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(tfidf_path, 'rb') as f:
                self.tfidf_matrix = pickle.load(f)
            self.embeddings = np.load(embeddings_path)

            # Initialize the sentence transformer model (needed for encoding new queries)
            print("Loading sentence transformer model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

            print(f" Models loaded from {directory}/")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def save_models(self, directory='models'):
        os.makedirs(directory, exist_ok=True)

        with open(f'{directory}/tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(f'{directory}/album_tfidf_matrix.pkl', 'wb') as f:
            pickle.dump(self.tfidf_matrix, f)
        np.save(f'{directory}/album_semantic_embeddings.npy', self.embeddings)

        print(f" Models saved to {directory}/ directory.")


def main():
    print("Album Recommendation System initializing...")
    print("Building models: TF-IDF (text similarity), semantic embeddings (contextual meaning), and feature-engineered vectors (album attributes)...")

    recommender = EnhancedRecommender()
    recommender.build_models()
    recommender.save_models()

    print("Album Recommendation Models built and saved.")


if __name__ == "__main__":
    main()