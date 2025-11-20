import pandas as pd
from transformers import pipeline
import re
from collections import Counter
import warnings

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
            
            positive_words = ['brilliant', 'excellent', 'masterpiece', 'stunning', 'beautiful', 'powerful', 
                            'impressive', 'best', 'perfect', 'remarkable', 'extraordinary', 'phenomenal', 
                            'incredible', 'outstanding', 'exceptional', 'captivating', 'mesmerizing', 
                            'triumphant', 'transcendent', 'sublime', 'magnificent', 'superb', 'spectacular',
                            'gorgeous', 'exquisite', 'glorious', 'breathtaking', 'amazing', 'fantastic',
                            'wonderful', 'terrific', 'marvelous', 'brilliant', 'splendid', 'dazzling',
                            'luminous', 'radiant', 'vibrant', 'vivid', 'compelling', 'engaging',
                            'riveting', 'enthralling', 'gripping', 'fascinating', 'intriguing']
            
            negative_words = ['disappointing', 'weak', 'boring', 'forgettable', 'mediocre', 'poor', 
                            'worst', 'fails', 'lacking', 'dull', 'uninspired', 'lifeless', 'bland', 
                            'tedious', 'frustrating', 'hollow', 'empty', 'derivative', 'flat', 'stale',
                            'lackluster', 'unremarkable', 'pedestrian', 'mundane', 'insipid', 'vapid',
                            'trite', 'banal', 'clumsy', 'awkward', 'clunky', 'messy', 'sloppy',
                            'amateurish', 'half-baked', 'undercooked', 'overcooked', 'uneven',
                            'inconsistent', 'disjointed', 'confused', 'muddled', 'misguided']
            
            musical_terms = ['beat', 'melody', 'production', 'vocal', 'instrumental', 'rhythm', 'bass', 
                           'guitar', 'drums', 'sound', 'track', 'arrangement', 'composition', 'performance',
                           'harmony', 'chord', 'tempo', 'groove', 'riff', 'synthesizer', 'mixing', 
                           'mastering', 'verse', 'chorus', 'bridge', 'outro', 'intro', 'hook', 'melody',
                           'percussion', 'instrumentation', 'timbre', 'texture', 'dynamics', 'crescendo',
                           'diminuendo', 'cadence', 'phrasing', 'articulation', 'vibrato', 'sustain',
                           'reverb', 'delay', 'distortion', 'compression', 'eq', 'equalization']
            
            descriptive_words = ['atmospheric', 'raw', 'polished', 'experimental', 'nostalgic', 'modern',
                               'minimalist', 'dense', 'intimate', 'grandiose', 'subtle', 'bold', 'delicate',
                               'aggressive', 'gentle', 'dynamic', 'static', 'organic', 'synthetic', 'warm',
                               'cold', 'bright', 'dark', 'light', 'heavy', 'smooth', 'rough', 'crisp',
                               'fuzzy', 'sharp', 'soft', 'hard', 'clean', 'dirty', 'dry', 'wet',
                               'thick', 'thin', 'tight', 'loose', 'punchy', 'muddy', 'clear']
            
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
        
        intensity_modifiers = {
            'very': 2, 'extremely': 2, 'incredibly': 2, 'exceptionally': 2,
            'highly': 1.5, 'quite': 1.3, 'fairly': 1.2,
            'somewhat': 0.8, 'slightly': 0.7, 'barely': 0.5
        }
        
        char_patterns = {
            'tempo_fast': ['fast', 'rapid', 'quick', 'brisk', 'frenetic', 'breakneck'],
            'tempo_slow': ['slow', 'languid', 'leisurely', 'crawling', 'glacial'],
            'volume_loud': ['loud', 'thunderous', 'booming', 'deafening', 'explosive'],
            'volume_quiet': ['quiet', 'soft', 'whisper', 'hushed', 'delicate'],
            'complexity_high': ['complex', 'intricate', 'layered', 'multifaceted', 'sophisticated'],
            'complexity_low': ['simple', 'straightforward', 'minimalist', 'sparse', 'bare'],
            'texture_dense': ['dense', 'thick', 'heavy', 'lush', 'packed', 'wall of sound'],
            'texture_light': ['airy', 'spacious', 'light', 'breezy', 'sparse'],
        }
        
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
        
        instrument_keywords = {
            'guitar': ['guitar', 'guitars', 'riff', 'riffs', 'strum', 'strumming', 'fretwork', 'six-string'],
            'bass': ['bass', 'bassline', 'bass line', 'low end', 'bottom end', 'four-string', 'upright bass'],
            'drums': ['drums', 'drummer', 'percussion', 'beat', 'beats', 'snare', 'kick', 'hi-hat', 'cymbal', 'tom'],
            'piano': ['piano', 'keys', 'keyboard', 'ivories', 'grand piano', 'upright piano', 'electric piano'],
            'synthesizer': ['synth', 'synthesizer', 'synths', 'synthesizers', 'moog', 'analog synth', 'modular', 'arp'],
            'vocals': ['vocals', 'voice', 'singing', 'singer', 'sung', 'vocalization', 'harmonies', 'falsetto', 'crooning'],
            'strings': ['strings', 'violin', 'viola', 'cello', 'orchestra', 'orchestral', 'string section', 'quartet'],
            'horns': ['horn', 'horns', 'trumpet', 'saxophone', 'sax', 'trombone', 'brass', 'brass section', 'clarinet'],
            'electronic': ['electronic', '808', '909', 'drum machine', 'sequencer', 'sampler', 'arpeggiator', 'vocoder'],
            'acoustic': ['acoustic', 'unplugged', 'live instruments', 'natural', 'organic sound'],
            'sampling': ['sample', 'samples', 'sampling', 'sampled', 'loop', 'loops', 'breakbeat', 'chopped']
        }
        
        for instrument, keywords in instrument_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                instruments.append(instrument)
        
        return instruments
    
    def extract_production_quality(self, text):
        """Extract production quality indicators"""
        if not text:
            return {'quality': 'unknown', 'style': 'unknown'}
        
        text_lower = text.lower()
        
        quality_indicators = {
            'polished': ['polished', 'pristine', 'professional', 'slick', 'hi-fi', 'clean production'],
            'raw': ['raw', 'rough', 'unpolished', 'lo-fi', 'DIY', 'home recorded', 'bedroom'],
            'experimental': ['experimental production', 'unconventional production', 'avant-garde production'],
        }
        
        style_indicators = {
            'analog': ['analog', 'analogue', 'tape', 'vintage gear', 'warm'],
            'digital': ['digital', 'computer', 'DAW', 'plugin', 'modern production'],
            'minimalist': ['minimal production', 'sparse production', 'stripped down'],
            'maximalist': ['wall of sound', 'dense production', 'layered production', 'lush production']
        }
        
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
        
        mood_keywords = {
            'sad': ['sad', 'melancholy', 'somber', 'mournful', 'depressing', 'gloomy', 'sorrowful', 'dejected', 'despondent', 'forlorn'],
            'happy': ['happy', 'joyful', 'cheerful', 'upbeat', 'celebratory', 'euphoric', 'elated', 'ecstatic', 'jubilant', 'exuberant'],
            'angry': ['angry', 'aggressive', 'furious', 'hostile', 'violent', 'rage', 'wrathful', 'indignant', 'incensed', 'livid'],
            'calm': ['calm', 'peaceful', 'serene', 'tranquil', 'soothing', 'gentle', 'placid', 'still', 'quiet', 'restful'],
            'anxious': ['anxious', 'tense', 'nervous', 'uneasy', 'restless', 'paranoid', 'edgy', 'jittery', 'worried', 'fraught'],
            'romantic': ['romantic', 'intimate', 'sensual', 'loving', 'tender', 'affectionate', 'passionate', 'amorous'],
            'dark': ['dark', 'sinister', 'ominous', 'foreboding', 'menacing', 'bleak', 'grim', 'shadowy', 'murky', 'eerie'],
            'dreamy': ['dreamy', 'ethereal', 'hazy', 'floating', 'otherworldly', 'surreal', 'hypnotic', 'trance-like', 'mystical'],
            'melancholic': ['melancholic', 'wistful', 'pensive', 'reflective', 'contemplative', 'introspective', 'nostalgic'],
            'uplifting': ['uplifting', 'inspiring', 'hopeful', 'optimistic', 'encouraging', 'motivating', 'empowering']
        }
        
        energy_indicators = {
            'high': ['energetic', 'explosive', 'frenetic', 'intense', 'powerful', 'driving', 'relentless'],
            'medium': ['moderate', 'steady', 'consistent', 'balanced'],
            'low': ['slow', 'subdued', 'mellow', 'laid-back', 'downtempo', 'languid']
        }
        
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
        polarizing_phrases = [
            'love it or hate it', 'divisive', 'polarizing', 'marmite',
            'not for everyone', 'challenging listen', 'acquired taste',
            'controversial', 'some will love', 'some will hate',
            'won\'t appeal to everyone', 'turn off some listeners'
        ]
        
        return any(phrase in text_lower for phrase in polarizing_phrases)
    
    def extract_novelty_indicators(self, text):
        """Extract novelty vs. derivative indicators"""
        if not text:
            return 0  # 0 = neutral
        
        text_lower = text.lower()
        
        novelty_positive = [
            'innovative', 'groundbreaking', 'fresh', 'original', 'unique',
            'pioneering', 'trailblazing', 'inventive', 'creative', 'novel',
            'revolutionary', 'game-changer', 'unprecedented'
        ]
        
        novelty_negative = [
            'derivative', 'formulaic', 'predictable', 'generic', 'clichéd',
            'unoriginal', 'by-the-numbers', 'paint-by-numbers', 'cookie-cutter',
            'rehash', 'retread', 'stale', 'tired'
        ]
        
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
        
        era_keywords = {
            'vintage': ['vintage', 'retro', 'throwback', 'nostalgic', 'classic sound'],
            'contemporary': ['modern', 'contemporary', 'current', 'today', 'now'],
            'timeless': ['timeless', 'ageless', 'eternal', 'enduring']
        }
        
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
        
        context_keywords = {
            'party': ['party', 'club', 'dancefloor', 'dancing', 'celebration'],
            'study': ['focus', 'concentration', 'background', 'ambient', 'work'],
            'workout': ['energetic', 'pump', 'gym', 'exercise', 'adrenaline'],
            'relaxation': ['relax', 'chill', 'unwind', 'calm', 'soothing', 'meditative'],
            'driving': ['driving', 'road trip', 'cruising', 'highway'],
            'intimate': ['intimate', 'bedroom', 'late night', 'romantic setting'],
            'headphones': ['headphone', 'headphones', 'close listening', 'detailed listening']
        }
        
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
        
        theme_keywords = {
            'love_romance': ['love', 'romance', 'relationship', 'heartbreak', 'longing', 'desire', 'affection', 'devotion', 'infatuation', 'crush'],
            'social_political': ['political', 'protest', 'activism', 'justice', 'inequality', 'revolution', 'oppression', 'freedom', 'rights', 'resistance'],
            'personal_growth': ['growth', 'self-discovery', 'identity', 'coming-of-age', 'maturity', 'transformation', 'evolution', 'awakening'],
            'existential': ['existential', 'mortality', 'death', 'meaning', 'purpose', 'existence', 'nihilism', 'void', 'absurdity'],
            'urban_life': ['city', 'urban', 'street', 'neighborhood', 'metropolitan', 'downtown', 'cityscape', 'concrete jungle'],
            'nature': ['nature', 'landscape', 'environment', 'earth', 'wilderness', 'forest', 'ocean', 'mountains', 'sky', 'seasons'],
            'nostalgia_memory': ['memory', 'past', 'nostalgia', 'reminisce', 'childhood', 'youth', 'remembrance', 'bygone', 'yesterday'],
            'struggle_hardship': ['struggle', 'hardship', 'pain', 'suffering', 'adversity', 'challenge', 'obstacle', 'burden', 'trial'],
            'celebration': ['celebration', 'joy', 'party', 'triumph', 'victory', 'success', 'achievement', 'glory', 'festivity'],
            'spirituality': ['spiritual', 'faith', 'divine', 'transcendent', 'mystical', 'sacred', 'holy', 'prayer', 'meditation', 'enlightenment'],
            'isolation': ['isolation', 'loneliness', 'solitude', 'alienation', 'disconnection', 'abandonment', 'forsaken', 'outcast'],
            'desire_lust': ['desire', 'lust', 'craving', 'yearning', 'hunger', 'temptation', 'seduction', 'passion'],
            'rebellion': ['rebellion', 'defiance', 'revolt', 'insurgency', 'nonconformity', 'dissent', 'mutiny']
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                lyrical_themes.append(theme)
        
        return lyrical_themes
    
    def extract_themes(self, text):
        if not text:
            return []
        
        text_lower = text.lower()
        
        theme_keywords = {
            'production_pristine': ['pristine', 'polished', 'clean', 'crisp', 'clarity', 'hi-fi', 'slick'],
            'production_experimental': ['production', 'produced', 'sound design', 'sonic', 'mixing', 'mastering', 'layering'],
            'vocals_powerful': ['powerful vocals', 'soaring', 'belting', 'commanding voice', 'vocal prowess'],
            'vocals_intimate': ['intimate vocals', 'whisper', 'breathy', 'tender voice', 'delicate singing'],
            'vocals_harmonies': ['harmonies', 'vocal layers', 'backing vocals', 'choir', 'multitracked'],
            'lyrics_poetic': ['poetic', 'wordplay', 'metaphor', 'lyrical', 'verse', 'poetry'],
            'lyrics_narrative': ['storytelling', 'narrative', 'story', 'character', 'plot', 'tale'],
            'emotion_melancholic': ['melancholy', 'sad', 'somber', 'mournful', 'wistful', 'sorrowful'],
            'emotion_joyful': ['joyful', 'happy', 'cheerful', 'euphoric', 'blissful', 'elated'],
            'emotion_angry': ['angry', 'rage', 'furious', 'hostile', 'aggressive emotion'],
            'emotion_vulnerable': ['vulnerable', 'fragile', 'exposed', 'heartbreak', 'raw emotion'],
            'experimental_avant': ['experimental', 'avant-garde', 'unconventional', 'boundary-pushing', 'radical'],
            'experimental_innovative': ['innovative', 'groundbreaking', 'pioneering', 'trailblazing', 'revolutionary'],
            'energy_explosive': ['explosive', 'relentless', 'frenetic', 'ferocious', 'breakneck'],
            'energy_driving': ['driving', 'propulsive', 'kinetic', 'momentum', 'pulsing'],
            'energy_chill': ['chill', 'relaxed', 'laid-back', 'mellow', 'easygoing', 'downtempo'],
            'atmosphere_dreamy': ['dreamy', 'ethereal', 'otherworldly', 'floating', 'weightless'],
            'atmosphere_dark': ['dark', 'brooding', 'ominous', 'menacing', 'sinister', 'foreboding'],
            'atmosphere_spacious': ['spacious', 'expansive', 'vast', 'open', 'airy'],
            'atmosphere_claustrophobic': ['claustrophobic', 'suffocating', 'dense', 'oppressive'],
            'atmosphere_cinematic': ['cinematic', 'soundtrack', 'film score', 'orchestral sweep'],
            'rhythm_syncopated': ['syncopated', 'off-kilter', 'complex rhythm', 'polyrhythm'],
            'rhythm_hypnotic': ['hypnotic', 'repetitive', 'trance', 'cyclical', 'looping'],
            'rhythm_groovy': ['groovy', 'funky', 'swagger', 'bounce', 'pocket'],
            'texture_lush': ['lush', 'rich', 'thick', 'full', 'abundant'],
            'texture_sparse': ['sparse', 'minimal', 'bare', 'stripped', 'skeletal'],
            'genre_fusion': ['fusion', 'blend', 'hybrid', 'cross-genre', 'mixing genres'],
            'nostalgic_retro': ['nostalgic', 'throwback', 'retro', 'vintage', '70s', '80s', '90s'],
            'nostalgic_timeless': ['timeless', 'classic', 'enduring', 'ageless'],
            'danceable': ['danceable', 'club', 'dancefloor', 'party', 'move'],
            'catchy': ['catchy', 'hook', 'memorable', 'infectious', 'earworm', 'anthemic'],
            'technical_virtuosic': ['virtuosic', 'technical mastery', 'precision', 'skillful', 'dexterity', 'prowess', 'expert'],
            'technical_complex': ['complex', 'intricate', 'sophisticated', 'multifaceted', 'elaborate', 'ornate'],
            'technical_virtuosic': ['virtuosic', 'technical mastery', 'precision', 'skillful', 'dexterity', 'prowess', 'expert'],
            'technical_complex': ['complex', 'intricate', 'sophisticated', 'multifaceted', 'elaborate', 'ornate'],
            'raw_gritty': ['gritty', 'rough', 'unpolished', 'lo-fi', 'DIY', 'rough-hewn', 'coarse'],
            'raw_visceral': ['visceral', 'primal', 'intense', 'immediate', 'gut-wrenching', 'powerful'],
            'uplifting': ['uplifting', 'hopeful', 'optimistic', 'triumphant', 'inspiring', 'elevating', 'soaring'],
            'introspective': ['introspective', 'reflective', 'contemplative', 'meditative', 'philosophical', 'thoughtful'],
            'playful': ['playful', 'whimsical', 'quirky', 'tongue-in-cheek', 'irreverent', 'cheeky', 'mischievous'],
            'sensual': ['sensual', 'sexy', 'sultry', 'seductive', 'erotic', 'steamy', 'provocative'],
            'psychedelic': ['psychedelic', 'trippy', 'mind-bending', 'hallucinogenic', 'kaleidoscopic', 'lysergic'],
            'abrasive': ['abrasive', 'harsh', 'confrontational', 'challenging', 'jarring', 'grating', 'caustic'],
            'beautiful': ['beautiful', 'gorgeous', 'stunning', 'exquisite', 'lovely', 'sublime', 'magnificent'],
            'chaotic': ['chaotic', 'frantic', 'messy', 'dissonant', 'turbulent', 'frenetic', 'wild', 'anarchic'],
            'cohesive': ['cohesive', 'unified', 'consistent', 'focused', 'tight', 'seamless', 'integrated'],
            'raw_gritty': ['gritty', 'rough', 'unpolished', 'lo-fi', 'DIY', 'rough-hewn', 'coarse'],
            'raw_visceral': ['visceral', 'primal', 'intense', 'immediate', 'gut-wrenching', 'powerful'],
            'uplifting': ['uplifting', 'hopeful', 'optimistic', 'triumphant', 'inspiring', 'elevating', 'soaring'],
            'introspective': ['introspective', 'reflective', 'contemplative', 'meditative', 'philosophical', 'thoughtful'],
            'playful': ['playful', 'whimsical', 'quirky', 'tongue-in-cheek', 'irreverent', 'cheeky', 'mischievous'],
            'sensual': ['sensual', 'sexy', 'sultry', 'seductive', 'erotic', 'steamy', 'provocative'],
            'psychedelic': ['psychedelic', 'trippy', 'mind-bending', 'hallucinogenic', 'kaleidoscopic', 'lysergic'],
            'abrasive': ['abrasive', 'harsh', 'confrontational', 'challenging', 'jarring', 'grating', 'caustic'],
            'beautiful': ['beautiful', 'gorgeous', 'stunning', 'exquisite', 'lovely', 'sublime', 'magnificent'],
            'chaotic': ['chaotic', 'frantic', 'messy', 'dissonant', 'turbulent', 'frenetic', 'wild', 'anarchic'],
            'cohesive': ['cohesive', 'unified', 'consistent', 'focused', 'tight', 'seamless', 'integrated'],
            'ambient': ['ambient', 'environmental', 'soundscape', 'textural', 'immersive', 'enveloping'],
            'confrontational': ['confrontational', 'provocative', 'challenging', 'unsettling', 'discomforting'],
            'euphoric': ['euphoric', 'ecstatic', 'blissful', 'transcendent', 'rapturous', 'exultant'],
            'haunting': ['haunting', 'ghostly', 'spectral', 'eerie', 'uncanny', 'unsettling', 'lingering'],
            'majestic': ['majestic', 'grand', 'epic', 'monumental', 'towering', 'imposing', 'regal'],
            'vulnerable': ['vulnerable', 'exposed', 'bare', 'unguarded', 'open', 'confessional']
        }
        
        themes = []
        for theme, keywords in theme_keywords.items():
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