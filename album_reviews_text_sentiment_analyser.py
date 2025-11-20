import pandas as pd
from transformers import pipeline
import re
from collections import Counter
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import logging

logging.set_verbosity_error()

# from the sentiment directory, retrieve the following sentiment matching keywords rules;
from sentiment.sentiment_keywords import (
    positive_words, negative_words, musical_terms, descriptive_words,
    intensity_modifiers, char_patterns, instrument_keywords, quality_indicators,
    style_indicators, mood_keywords, energy_indicators, polarizing_phrases,
    novelty_positive, novelty_negative, context_keywords, era_keywords, lyrical_theme_keywords, general_theme_keywords
)


class ReviewAnalyser:
    def __init__(self, data_path='outputs/pitchfork_reviews_preprocessed.csv'):
        # Loading silently
        self.df = pd.read_csv(data_path)
        self.sentiment_analyzer = None
        self.summarizer = None

        # Validate that enhanced preprocessing columns exist
        if 'review_text_processed' not in self.df.columns:
            print(
                "Warning: 'review_text_processed' column not found. Using original 'review_text'.")
            self.df['review_text_processed'] = self.df['review_text']

    def load_models(self):
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1
        )
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1
        )
        # Models loaded silently

    def analyze_sentiment(self, text, max_length=512):
        if not text or len(text.strip()) == 0:
            return {'label': 'NEUTRAL', 'score': 0.5}

        truncated = text[:max_length]
        try:
            result = self.sentiment_analyzer(truncated)[0]
            return result
        except:
            return {'label': 'NEUTRAL', 'score': 0.5}

    def extract_key_sentences(self, text, num_sentences=3):
        if not text or len(text.strip()) == 0:
            return []

        # Fix common HTML artifacts and conjoined words
        # Add space before capital letters that follow lowercase (common in HTML parsing issues)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        # Fix missing spaces after punctuation
        text = re.sub(r'([.!?,;:])([A-Za-z])', r'\1 \2', text)
        # Fix missing spaces around em dashes and similar
        text = re.sub(r'([a-z])—([a-z])', r'\1 — \2', text)

        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if len(sentences) == 0:
            return []

        scores = []
        for sentence in sentences:
            words = sentence.lower().split()

            score = 0
            score += len([w for w in words if w in positive_words]) * 3
            score += len([w for w in words if w in negative_words]) * 3
            score += len([w for w in words if w in musical_terms]) * 2
            score += len([w for w in words if w in descriptive_words]) * 2

            if any(word in sentence.lower() for word in ['because', 'however', 'although', 'while', 'whereas']):
                score += 2

            word_count = len(words)
            if 15 <= word_count <= 40:
                score += 3
            elif word_count > 40:
                score += 1

            scores.append(score)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :num_sentences]
        top_indices.sort()

        key_sentences = [sentences[i]
                         for i in top_indices if i < len(sentences)]
        return key_sentences

    def generate_summary(self, text, max_length=150, min_length=50):
        if not text or len(text.split()) < 50:
            return text[:200] if text else ""

        try:
            summary = self.summarizer(
                text[:1024],
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )

            return summary[0]['summary_text']

        except:
            return text[:200]

    def extract_musical_characteristics(self, text):
        """Extract specific musical characteristics with intensity scores"""
        if not text:
            return {}

        text_lower = text.lower()
        characteristics = {}

        for char, keywords in char_patterns.items():
            for keyword in keywords:
                if keyword in text_lower:
                    intensity = 1.0
                    for modifier, mult in intensity_modifiers.items():
                        if f"{modifier} {keyword}" in text_lower:
                            intensity = mult
                            break
                    characteristics[char] = max(
                        characteristics.get(char, 0), intensity)

        return characteristics

    def extract_instrumentation(self, text):
        """Extract mentioned instruments and sound sources"""
        if not text:
            return []

        text_lower = text.lower()
        instruments = []

        for instrument, keywords in instrument_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                instruments.append(instrument)

        return instruments

    def extract_production_quality(self, text):
        """Extract production quality indicators"""
        if not text:
            return {'quality': 'unknown', 'style': 'unknown'}

        text_lower = text.lower()

        quality = 'unknown'
        for q, keywords in quality_indicators.items():
            if any(keyword in text_lower for keyword in keywords):
                quality = q
                break

        style = 'unknown'
        for s, keywords in style_indicators.items():
            if any(keyword in text_lower for keyword in keywords):
                style = s
                break

        return {'quality': quality, 'style': style}

    def extract_mood_energy(self, text):
        """Extract mood and energy level indicators with scores"""
        if not text:
            return {'mood': [], 'energy': 0}

        text_lower = text.lower()

        moods = []
        for mood, keywords in mood_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                moods.append(mood)

        energy = 0  # -1 = low, 0 = medium, 1 = high
        if any(keyword in text_lower for keyword in energy_indicators['high']):
            energy = 1
        elif any(keyword in text_lower for keyword in energy_indicators['low']):
            energy = -1

        return {'mood': moods, 'energy': energy}

    def detect_polarizing_language(self, text):
        """Detect if album is described as polarizing or divisive"""
        if not text:
            return False

        text_lower = text.lower()

        return any(phrase in text_lower for phrase in polarizing_phrases)

    def extract_novelty_indicators(self, text):
        """Extract novelty vs. derivative indicators"""
        if not text:
            return 0  # 0 = neutral

        text_lower = text.lower()

        positive_count = sum(
            1 for word in novelty_positive if word in text_lower)
        negative_count = sum(
            1 for word in novelty_negative if word in text_lower)

        # Return score: positive = innovative, negative = derivative
        return positive_count - negative_count

    def analyze_critical_consensus(self, row):
        """Analyze alignment between score and sentiment"""
        score = float(row.get('score', 0))
        sentiment_label = row.get('sentiment_label', 'NEUTRAL')
        sentiment_score = float(row.get('sentiment_score', 0.5))

        # Expected sentiment based on score
        expected_positive = score >= 7.0
        actual_positive = sentiment_label == 'POSITIVE'

        # Check alignment
        aligned = expected_positive == actual_positive

        # Calculate consensus strength (how well score and sentiment agree)
        if aligned:
            consensus = 'strong'
        elif abs(score - 5.0) < 1.5:  # Mid-range scores allow disagreement
            consensus = 'mixed'
        else:
            consensus = 'conflicted'

        return consensus

    def extract_temporal_context(self, text, release_year):
        """Extract temporal context and era indicators"""
        if not text:
            return {'era_sound': 'contemporary', 'throwback': False}

        text_lower = text.lower()

        # Decade mentions
        decades = ['60s', '70s', '80s', '90s', '00s', '2000s',
                   'sixties', 'seventies', 'eighties', 'nineties']
        throwback = any(decade in text_lower for decade in decades)

        era_sound = 'contemporary'
        for era, keywords in era_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                era_sound = era
                break

        return {'era_sound': era_sound, 'throwback': throwback}

    def extract_context_indicators(self, text):
        """Extract listening context indicators"""
        if not text:
            return []

        text_lower = text.lower()
        contexts = []

        for context, keywords in context_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                contexts.append(context)

        return contexts

    def extract_comparative_context(self, text):
        """Extract artist comparisons and influences mentioned"""
        if not text:
            return []

        comparison_patterns = [
            r'(?:similar to|reminiscent of|echoes of|like|evokes|channels|recalls) ([A-Z][a-z]+(?: [A-Z][a-z]+)?)',
            r'([A-Z][a-z]+(?: [A-Z][a-z]+)?)-esque',
            r'(?:influence|inspired by|nods to) ([A-Z][a-z]+(?: [A-Z][a-z]+)?)',
        ]

        comparisons = []
        for pattern in comparison_patterns:
            matches = re.findall(pattern, text)
            comparisons.extend(matches)

        return list(set(comparisons))[:5]

    def extract_lyrical_themes(self, text):
        """Extract lyrical/content themes separate from musical themes"""
        if not text:
            return []

        text_lower = text.lower()
        lyrical_themes = []

        for theme, keywords in lyrical_theme_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                lyrical_themes.append(theme)

        return lyrical_themes

    def extract_themes(self, text):
        if not text:
            return []

        text_lower = text.lower()

        themes = []
        for theme, keywords in general_theme_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                themes.append(theme)

        return themes

    def analyze_all_reviews(self):
        print("Starting sentiment analysis for all album reviews...")
        df_sample = self.df
        text_column = 'review_text_processed' if 'review_text_processed' in df_sample.columns else 'review_text'


        def process_review(row):
            text_processed = str(row[text_column]) if pd.notna(row[text_column]) else ''
            text_original = str(row['review_text']) if pd.notna(row['review_text']) else ''
            sentiment = self.analyze_sentiment(text_processed)
            highlights = self.extract_key_sentences(text_original, num_sentences=4)
            themes = self.extract_themes(text_original)
            lyrical_themes = self.extract_lyrical_themes(text_original)
            comparisons = self.extract_comparative_context(text_original)
            musical_chars = self.extract_musical_characteristics(text_original)
            musical_chars_str = ', '.join([f"{k}:{v:.1f}" for k, v in musical_chars.items()])
            instruments = self.extract_instrumentation(text_original)
            prod_quality = self.extract_production_quality(text_original)
            mood_energy = self.extract_mood_energy(text_original)
            mood_str = ', '.join(mood_energy['mood']) if mood_energy['mood'] else 'neutral'
            energy_str = ['low', 'medium', 'high'][mood_energy['energy'] + 1]
            is_polarizing = self.detect_polarizing_language(text_original)
            novelty_score = self.extract_novelty_indicators(text_original)
            release_year = row.get('release_year', 2000)
            temporal = self.extract_temporal_context(text_original, release_year)
            contexts = self.extract_context_indicators(text_original)
            return {
                'sentiment': sentiment,
                'key_highlights': ' | '.join(highlights),
                'themes': ', '.join(themes) if themes else 'general',
                'lyrical_themes': ', '.join(lyrical_themes) if lyrical_themes else '',
                'comparisons': ', '.join(comparisons) if comparisons else '',
                'musical_chars': musical_chars_str,
                'instrumentation': ', '.join(instruments) if instruments else '',
                'production_quality': f"quality:{prod_quality['quality']}, style:{prod_quality['style']}",
                'mood_energy': f"{mood_str}, energy: {energy_str}",
                'is_polarizing': is_polarizing,
                'novelty_score': novelty_score,
                'temporal_context': f"era:{temporal['era_sound']}, throwback:{temporal['throwback']}",
                'listening_contexts': ', '.join(contexts) if contexts else ''
            }

        results = []

        num_workers = os.cpu_count() or 4
        # Parallel analysis starting (silent)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {executor.submit(process_review, row): idx for idx, row in df_sample.iterrows()}
            for i, future in enumerate(as_completed(future_to_idx)):
                idx = future_to_idx[future]
                # Only print every 500 reviews
                if i % 500 == 0:
                    print(f"Processed {i}/{len(df_sample)} album reviews sentiment...")
                try:
                    results.append((idx, future.result()))
                except Exception as e:
                    print(f"Error processing review {idx}: {e}")
                    results.append((idx, None))

        # Sort results by index to preserve order
        results.sort(key=lambda x: x[0])
        sentiments = [r[1]['sentiment'] if r[1] else {'label': 'NEUTRAL', 'score': 0.5} for r in results]
        key_highlights = [r[1]['key_highlights'] if r[1] else '' for r in results]
        themes_list = [r[1]['themes'] if r[1] else 'general' for r in results]
        lyrical_themes_list = [r[1]['lyrical_themes'] if r[1] else '' for r in results]
        comparisons_list = [r[1]['comparisons'] if r[1] else '' for r in results]
        musical_chars_list = [r[1]['musical_chars'] if r[1] else '' for r in results]
        instrumentation_list = [r[1]['instrumentation'] if r[1] else '' for r in results]
        production_quality_list = [r[1]['production_quality'] if r[1] else '' for r in results]
        mood_energy_list = [r[1]['mood_energy'] if r[1] else '' for r in results]
        polarizing_list = [r[1]['is_polarizing'] if r[1] else False for r in results]
        novelty_scores = [r[1]['novelty_score'] if r[1] else 0 for r in results]
        temporal_context_list = [r[1]['temporal_context'] if r[1] else '' for r in results]
        context_indicators_list = [r[1]['listening_contexts'] if r[1] else '' for r in results]

        print(f"\n✓ Sentiment analysis complete for {len(df_sample)} reviews.")

        # Existing columns
        df_sample['sentiment_label'] = [s['label'] for s in sentiments]
        df_sample['sentiment_score'] = [s['score'] for s in sentiments]
        df_sample['key_highlights'] = key_highlights
        df_sample['themes'] = themes_list
        df_sample['lyrical_themes'] = lyrical_themes_list
        df_sample['comparisons'] = comparisons_list
        df_sample['musical_characteristics'] = musical_chars_list

        # NEW COLUMNS
        df_sample['instrumentation'] = instrumentation_list
        df_sample['production_quality'] = production_quality_list
        df_sample['mood_energy'] = mood_energy_list
        df_sample['is_polarizing'] = polarizing_list
        df_sample['novelty_score'] = novelty_scores
        df_sample['temporal_context'] = temporal_context_list
        df_sample['listening_contexts'] = context_indicators_list

        # Calculate critical consensus
        df_sample['critical_consensus'] = df_sample.apply(
            self.analyze_critical_consensus, axis=1)

        # Print only summary stats
        print("\nSummary of sentiment analysis results:")
        print("Sentiment label distribution:")
        print(df_sample['sentiment_label'].value_counts())
        if 'score_category' in df_sample.columns:
            print("Average sentiment score by album rating category:")
            print(df_sample.groupby('score_category')['sentiment_score'].mean())
        print(f"\nPolarizing/Divisive Albums: {df_sample['is_polarizing'].sum()} ({df_sample['is_polarizing'].sum()/len(df_sample)*100:.1f}%)")
        print(f"\nNovelty - Innovative: {(df_sample['novelty_score'] > 0).sum()}, Derivative: {(df_sample['novelty_score'] < 0).sum()}, Neutral: {(df_sample['novelty_score'] == 0).sum()}")
        print("\nSaving sentiment analysed pitchfork reviews dataset...")
        os.makedirs('outputs', exist_ok=True)
        df_sample.to_csv(
            'outputs/pitchfork_reviews_preprocessed_plus_sentiments.csv', index=False)
        print("✓ Saved to: outputs/pitchfork_reviews_preprocessed_plus_sentiments.csv")

        return df_sample

    def show_examples(self, num_examples=3):
        print("\n" + "="*80)
        print("EXAMPLE ANALYZED ALBUM REVIEWS")
        print("="*80 + "\n")

        df_analyzed = pd.read_csv(
            'outputs/pitchfork_reviews_preprocessed_plus_sentiments.csv')

        samples = df_analyzed.sample(num_examples)

        for idx, row in samples.iterrows():
            print("-"*80)
            print(f"Album: {row['album_name']}")
            print(f"Artist: {row['artist_name']}")
            print(f"Genre: {row['genre']}")
            print(f"Score: {row['score']}")
            print(
                f"\nSentiment: {row['sentiment_label']} (confidence: {row['sentiment_score']:.2f})")
            print(f"Themes: {row['themes']}")
            print(f"\nKey Highlights:")
            highlights = row['key_highlights'].split(' | ')
            for i, highlight in enumerate(highlights, 1):
                print(f"  {i}. {highlight}")
            print(f"\nFull Review URL: {row['url']}")
            print()

        print("="*80 + "\n")


def main():
    # Use outputs directory for all files
    data_path = 'outputs/pitchfork_reviews_preprocessed.csv'
    output_path = 'outputs/pitchfork_reviews_preprocessed_plus_sentiments.csv'

    analyzer = ReviewAnalyser(data_path)
    analyzer.load_models()
    analyzer.analyze_all_reviews()

    print(f"✓ Complete! All albums sentiment analyzed and saved to '{output_path}'")

    analyzer.show_examples(num_examples=5)


if __name__ == "__main__":
    main()
