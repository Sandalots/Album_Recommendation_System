import pandas as pd
from transformers import pipeline
import re
from collections import Counter
import warnings

# from the sentiment directory, retrieve the following sentiment matching keywords rules;
from sentiment.sentiment_keywords import (
    positive_words, negative_words, musical_terms, descriptive_words,
    intensity_modifiers, char_patterns, instrument_keywords, quality_indicators,
    style_indicators, mood_keywords, energy_indicators, polarizing_phrases,
    novelty_positive, novelty_negative, context_keywords, era_keywords, lyrical_theme_keywords, general_theme_keywords
)

warnings.filterwarnings('ignore')

class ReviewAnalyser:
    def __init__(self, data_path='outputs/pitchfork_reviews_preprocessed.csv'):
        print("Loading album reviews dataset for sentiment analysis...")
        self.df = pd.read_csv(data_path)
        self.sentiment_analyzer = None
        self.summarizer = None
        
        # Validate that enhanced preprocessing columns exist
        if 'review_text_processed' not in self.df.columns:
            print("Warning: 'review_text_processed' column not found. Using original 'review_text'.")
            self.df['review_text_processed'] = self.df['review_text']
        
    def load_models(self):
        print("\nLoading model for album review sentiment analysis...")
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1
        )
        
        print("Loading text summarization model...")
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1
        )
        
        print("✓ Sentiment and summarization models loaded for album reviews\n")
    
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
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:num_sentences]
        top_indices.sort()
        
        key_sentences = [sentences[i] for i in top_indices if i < len(sentences)]
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
                    characteristics[char] = max(characteristics.get(char, 0), intensity)
        
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
        
        
        positive_count = sum(1 for word in novelty_positive if word in text_lower)
        negative_count = sum(1 for word in novelty_negative if word in text_lower)
        
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
        print("="*80)
        print("ALBUM REVIEWS SENTIMENT ANALYSIS")
        print("="*80 + "\n")
        
        df_sample = self.df
        print(f"Analyzing sentiment for all {len(df_sample)} album reviews...\n")
        
        # Use preprocessed text for efficiency (lemmatized version)
        text_column = 'review_text_processed' if 'review_text_processed' in df_sample.columns else 'review_text'
        print(f"Using '{text_column}' column for sentiment analysis (preprocessed text)\n")
        
        sentiments = []
        key_highlights = []
        summaries = []
        themes_list = []
        lyrical_themes_list = []
        comparisons_list = []
        musical_chars_list = []
        instrumentation_list = []
        production_quality_list = []
        mood_energy_list = []
        polarizing_list = []
        novelty_scores = []
        temporal_context_list = []
        context_indicators_list = []
        
        for idx, row in df_sample.iterrows():
            if idx % 50 == 0:
                print(f"  Sentiment processed for {idx}/{len(df_sample)} album reviews...")
            
            # Use preprocessed text for sentiment analysis (more accurate)
            text_processed = str(row[text_column]) if pd.notna(row[text_column]) else ''
            # Use original text for highlights extraction (more readable)
            text_original = str(row['review_text']) if pd.notna(row['review_text']) else ''
            
            # Sentiment on preprocessed text (better accuracy)
            sentiment = self.analyze_sentiment(text_processed)
            sentiments.append(sentiment)
            
            # Highlights from original text (more readable)
            highlights = self.extract_key_sentences(text_original, num_sentences=4)
            key_highlights.append(' | '.join(highlights))
            
            # Theme extraction from original text (needs context)
            themes = self.extract_themes(text_original)
            themes_list.append(', '.join(themes) if themes else 'general')
            
            lyrical_themes = self.extract_lyrical_themes(text_original)
            lyrical_themes_list.append(', '.join(lyrical_themes) if lyrical_themes else '')
            
            comparisons = self.extract_comparative_context(text_original)
            comparisons_list.append(', '.join(comparisons) if comparisons else '')
            
            musical_chars = self.extract_musical_characteristics(text_original)
            musical_chars_str = ', '.join([f"{k}:{v:.1f}" for k, v in musical_chars.items()])
            musical_chars_list.append(musical_chars_str)
            
            # NEW ANALYSES
            # Instrumentation
            instruments = self.extract_instrumentation(text_original)
            instrumentation_list.append(', '.join(instruments) if instruments else '')
            
            # Production quality
            prod_quality = self.extract_production_quality(text_original)
            production_quality_list.append(f"quality:{prod_quality['quality']}, style:{prod_quality['style']}")
            
            # Mood and energy
            mood_energy = self.extract_mood_energy(text_original)
            mood_str = ', '.join(mood_energy['mood']) if mood_energy['mood'] else 'neutral'
            energy_str = ['low', 'medium', 'high'][mood_energy['energy'] + 1]
            mood_energy_list.append(f"{mood_str}, energy: {energy_str}")
            
            # Polarizing language
            is_polarizing = self.detect_polarizing_language(text_original)
            polarizing_list.append(is_polarizing)
            
            # Novelty indicators
            novelty_score = self.extract_novelty_indicators(text_original)
            novelty_scores.append(novelty_score)
            
            # Temporal context
            release_year = row.get('release_year', 2000)
            temporal = self.extract_temporal_context(text_original, release_year)
            temporal_context_list.append(f"era:{temporal['era_sound']}, throwback:{temporal['throwback']}")
            
            # Context indicators
            contexts = self.extract_context_indicators(text_original)
            context_indicators_list.append(', '.join(contexts) if contexts else '')
        
        print(f"✓ Completed sentiment analysis for {len(df_sample)} album reviews\n")
        
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
        df_sample['critical_consensus'] = df_sample.apply(self.analyze_critical_consensus, axis=1)
        
        print("="*80)
        print("ALBUM REVIEWS SENTIMENT ANALYSIS RESULTS")
        print("="*80)
        print(f"\nAlbum Review Sentiment Distribution:")
        print(df_sample['sentiment_label'].value_counts())
        
        print(f"\nAverage Sentiment Score by Album Rating Category:")
        if 'score_category' in df_sample.columns:
            sentiment_by_score = df_sample.groupby('score_category')['sentiment_score'].mean()
            print(sentiment_by_score)
        
        print("\n" + "="*80)
        print("THEME ANALYSIS")
        print("="*80)
        all_themes = []
        for theme_str in df_sample['themes']:
            all_themes.extend(theme_str.split(', '))
        
        theme_counts = Counter(all_themes)
        print(f"\nMost Common Themes:")
        for theme, count in theme_counts.most_common(10):
            print(f"  {theme}: {count}")
        
        print("\n" + "="*80)
        print("ENHANCED FEATURE ANALYSIS")
        print("="*80)
        
        # Instrumentation analysis
        all_instruments = []
        for instr_str in df_sample['instrumentation']:
            if instr_str:
                all_instruments.extend(instr_str.split(', '))
        instrument_counts = Counter(all_instruments)
        print(f"\nMost Common Instruments:")
        for instrument, count in instrument_counts.most_common(10):
            print(f"  {instrument}: {count}")
        
        # Polarizing albums
        polarizing_count = df_sample['is_polarizing'].sum()
        print(f"\nPolarizing/Divisive Albums: {polarizing_count} ({polarizing_count/len(df_sample)*100:.1f}%)")
        
        # Novelty distribution
        print(f"\nNovelty Score Distribution:")
        print(f"  Innovative (>0): {(df_sample['novelty_score'] > 0).sum()}")
        print(f"  Derivative (<0): {(df_sample['novelty_score'] < 0).sum()}")
        print(f"  Neutral (=0): {(df_sample['novelty_score'] == 0).sum()}")
        
        # Critical consensus
        print(f"\nCritical Consensus:")
        print(df_sample['critical_consensus'].value_counts())
        
        # Listening contexts
        all_contexts = []
        for context_str in df_sample['listening_contexts']:
            if context_str:
                all_contexts.extend(context_str.split(', '))
        context_counts = Counter(all_contexts)
        print(f"\nListening Contexts:")
        for context, count in context_counts.most_common():
            print(f"  {context}: {count}")
        
        print("\n" + "="*80)
        print("Saving enhanced dataset...")
        print("="*80)
        import os
        os.makedirs('outputs', exist_ok=True)
        df_sample.to_csv('outputs/pitchfork_reviews_preprocessed_plus_sentiments.csv', index=False)
        print("✓ Saved to: outputs/pitchfork_reviews_preprocessed_plus_sentiments.csv")
        
        return df_sample
    
    def show_examples(self, num_examples=3):
        print("\n" + "="*80)
        print("EXAMPLE ANALYZED REVIEWS")
        print("="*80 + "\n")
        
        df_analyzed = pd.read_csv('outputs/pitchfork_reviews_preprocessed_plus_sentiments.csv')
        
        samples = df_analyzed.sample(num_examples)
        
        for idx, row in samples.iterrows():
            print("-"*80)
            print(f"Album: {row['album_name']}")
            print(f"Artist: {row['artist_name']}")
            print(f"Genre: {row['genre']}")
            print(f"Score: {row['score']}")
            print(f"\nSentiment: {row['sentiment_label']} (confidence: {row['sentiment_score']:.2f})")
            print(f"Themes: {row['themes']}")
            print(f"\nKey Highlights:")
            highlights = row['key_highlights'].split(' | ')
            for i, highlight in enumerate(highlights, 1):
                print(f"  {i}. {highlight}")
            print(f"\nFull Review URL: {row['url']}")
            print()
        
        print("="*80 + "\n")


def main():
    print("Starting full dataset analysis...")
    print("This will analyze all albums in the dataset.\n")
    
    # Use outputs directory for all files
    data_path = 'outputs/pitchfork_reviews_preprocessed.csv'
    output_path = 'outputs/pitchfork_reviews_preprocessed_plus_sentiments.csv'
    
    analyzer = ReviewAnalyser(data_path)
    analyzer.load_models()
    analyzer.analyze_all_reviews()
    
    print(f"\n✓ Complete! All albums analyzed and saved to '{output_path}'")
    print("\n🎵 Enhanced with comprehensive features:")
    print("  • Instrumentation tracking")
    print("  • Production quality & style")
    print("  • Mood & energy levels")
    print("  • Polarizing language detection")
    print("  • Novelty indicators")
    print("  • Temporal context")
    print("  • Listening context suggestions")
    print("  • Critical consensus analysis")
    print("\nYou can now run the Streamlit app!")
    
    analyzer.show_examples(num_examples=5)

if __name__ == "__main__":
    main()