# Sentiment Keyword Lists and Dictionaries
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

polarizing_phrases = [
    'love it or hate it', 'divisive', 'polarizing', 'marmite',
    'not for everyone', 'challenging listen', 'acquired taste',
    'controversial', 'some will love', 'some will hate',
    'won\'t appeal to everyone', 'turn off some listeners'
]

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

context_keywords = {
    'party': ['party', 'club', 'dancefloor', 'dancing', 'celebration'],
    'study': ['focus', 'concentration', 'background', 'ambient', 'work'],
    'workout': ['energetic', 'pump', 'gym', 'exercise', 'adrenaline'],
    'relaxation': ['relax', 'chill', 'unwind', 'calm', 'soothing', 'meditative'],
    'driving': ['driving', 'road trip', 'cruising', 'highway'],
    'intimate': ['intimate', 'bedroom', 'late night', 'romantic setting'],
    'headphones': ['headphone', 'headphones', 'close listening', 'detailed listening']
}

era_keywords = {
    'vintage': ['vintage', 'retro', 'throwback', 'nostalgic', 'classic sound'],
    'contemporary': ['modern', 'contemporary', 'current', 'today', 'now'],
    'timeless': ['timeless', 'ageless', 'eternal', 'enduring']
}

lyrical_theme_keywords = {
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

general_theme_keywords = {
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
