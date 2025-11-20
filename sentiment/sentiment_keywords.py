# Sentiment Keyword Lists and Dictionaries
positive_words = ['brilliant', 'excellent', 'masterpiece', 'stunning', 'beautiful', 'powerful',
                  'impressive', 'best', 'perfect', 'remarkable', 'extraordinary', 'phenomenal',
                  'incredible', 'outstanding', 'exceptional', 'captivating', 'mesmerizing',
                  'triumphant', 'transcendent', 'sublime', 'magnificent', 'superb', 'spectacular',
                  'gorgeous', 'exquisite', 'glorious', 'breathtaking', 'amazing', 'fantastic',
                  'wonderful', 'terrific', 'marvelous', 'brilliant', 'splendid', 'dazzling',
                  'luminous', 'radiant', 'vibrant', 'vivid', 'compelling', 'engaging',
                  'riveting', 'enthralling', 'gripping', 'fascinating', 'intriguing',
                  'stellar', 'impeccable', 'flawless', 'enchanting', 'inspiring', 'admirable', 'acclaimed', 'lauded', 'renowned', 'distinguished']

negative_words = ['disappointing', 'weak', 'boring', 'forgettable', 'mediocre', 'poor',
                  'worst', 'fails', 'lacking', 'dull', 'uninspired', 'lifeless', 'bland',
                  'tedious', 'frustrating', 'hollow', 'empty', 'derivative', 'flat', 'stale',
                  'lackluster', 'unremarkable', 'pedestrian', 'mundane', 'insipid', 'vapid',
                  'trite', 'banal', 'clumsy', 'awkward', 'clunky', 'messy', 'sloppy',
                  'amateurish', 'half-baked', 'undercooked', 'overcooked', 'uneven',
                  'inconsistent', 'disjointed', 'confused', 'muddled', 'misguided',
                  'regrettable', 'unimpressive', 'unconvincing', 'unrefined', 'unpolished',
                  'unfocused', 'unpleasant', 'unappealing', 'unresolved', 'unbalanced']

musical_terms = ['beat', 'melody', 'production', 'vocal', 'instrumental', 'rhythm', 'bass',
                 'guitar', 'drums', 'sound', 'track', 'arrangement', 'composition', 'performance',
                 'harmony', 'chord', 'tempo', 'groove', 'riff', 'synthesizer', 'mixing',
                 'mastering', 'verse', 'chorus', 'bridge', 'outro', 'intro', 'hook', 'melody',
                 'percussion', 'instrumentation', 'timbre', 'texture', 'dynamics', 'crescendo',
                 'diminuendo', 'cadence', 'phrasing', 'articulation', 'vibrato', 'sustain',
                 'reverb', 'delay', 'distortion', 'compression', 'eq', 'equalization',
                 'modulation', 'octave', 'ensemble', 'solo', 'duet', 'trio', 'quartet', 'improvisation', 'syncopation', 'counterpoint']

descriptive_words = ['atmospheric', 'raw', 'polished', 'experimental', 'nostalgic', 'modern',
                     'minimalist', 'dense', 'intimate', 'grandiose', 'subtle', 'bold', 'delicate',
                     'aggressive', 'gentle', 'dynamic', 'static', 'organic', 'synthetic', 'warm',
                     'cold', 'bright', 'dark', 'light', 'heavy', 'smooth', 'rough', 'crisp',
                     'fuzzy', 'sharp', 'soft', 'hard', 'clean', 'dirty', 'dry', 'wet',
                     'thick', 'thin', 'tight', 'loose', 'punchy', 'muddy', 'clear',
                     'lush', 'airy', 'spacious', 'vivid', 'muted', 'vibrant', 'opaque', 'transparent', 'resonant', 'subdued']

intensity_modifiers = {
    'very': 2, 'extremely': 2, 'incredibly': 2, 'exceptionally': 2,
    'highly': 1.5, 'quite': 1.3, 'fairly': 1.2,
    'somewhat': 0.8, 'slightly': 0.7, 'barely': 0.5,
    'totally': 2, 'utterly': 2, 'moderately': 1.1, 'mildly': 0.9
}

char_patterns = {
    'tempo_fast': ['fast', 'rapid', 'quick', 'brisk', 'frenetic', 'breakneck', 'speedy', 'accelerated', 'zippy', 'swift', 'hasty'],
    'tempo_slow': ['slow', 'languid', 'leisurely', 'crawling', 'glacial', 'sluggish', 'unhurried', 'plodding', 'dragging', 'lazy'],
    'volume_loud': ['loud', 'thunderous', 'booming', 'deafening', 'explosive', 'blaring', 'piercing', 'roaring', 'resounding', 'ear-splitting'],
    'volume_quiet': ['quiet', 'soft', 'whisper', 'hushed', 'delicate', 'muted', 'subdued', 'gentle', 'low', 'faint'],
    'complexity_high': ['complex', 'intricate', 'layered', 'multifaceted', 'sophisticated', 'elaborate', 'ornate', 'detailed', 'complicated', 'advanced'],
    'complexity_low': ['simple', 'straightforward', 'minimalist', 'sparse', 'bare', 'basic', 'elemental', 'plain', 'unadorned', 'rudimentary'],
    'texture_dense': ['dense', 'thick', 'heavy', 'lush', 'packed', 'wall of sound', 'congested', 'saturated', 'overloaded', 'full'],
    'texture_light': ['airy', 'spacious', 'light', 'breezy', 'sparse', 'open', 'delicate', 'thin', 'weightless', 'feathery'],
}

instrument_keywords = {
    'guitar': ['guitar', 'guitars', 'riff', 'riffs', 'strum', 'strumming', 'fretwork', 'six-string', 'electric guitar', 'acoustic guitar', 'lead guitar', 'rhythm guitar'],
    'bass': ['bass', 'bassline', 'bass line', 'low end', 'bottom end', 'four-string', 'upright bass', 'fretless bass', 'electric bass', 'bass guitar', 'double bass'],
    'drums': ['drums', 'drummer', 'percussion', 'beat', 'beats', 'snare', 'kick', 'hi-hat', 'cymbal', 'tom', 'drum kit', 'drum set', 'sticks', 'rimshot'],
    'piano': ['piano', 'keys', 'keyboard', 'ivories', 'grand piano', 'upright piano', 'electric piano', 'synth piano', 'baby grand', 'concert piano'],
    'synthesizer': ['synth', 'synthesizer', 'synths', 'synthesizers', 'moog', 'analog synth', 'modular', 'arp', 'digital synth', 'polyphonic synth', 'monosynth'],
    'vocals': ['vocals', 'voice', 'singing', 'singer', 'sung', 'vocalization', 'harmonies', 'falsetto', 'crooning', 'lead vocals', 'backing vocals', 'choral'],
    'strings': ['strings', 'violin', 'viola', 'cello', 'orchestra', 'orchestral', 'string section', 'quartet', 'fiddle', 'bowed strings', 'ensemble'],
    'horns': ['horn', 'horns', 'trumpet', 'saxophone', 'sax', 'trombone', 'brass', 'brass section', 'clarinet', 'flugelhorn', 'cornet', 'baritone'],
    'electronic': ['electronic', '808', '909', 'drum machine', 'sequencer', 'sampler', 'arpeggiator', 'vocoder', 'synth pad', 'electro', 'EDM', 'modulator'],
    'acoustic': ['acoustic', 'unplugged', 'live instruments', 'natural', 'organic sound', 'acoustic session', 'acoustic set', 'acoustic arrangement'],
    'sampling': ['sample', 'samples', 'sampling', 'sampled', 'loop', 'loops', 'breakbeat', 'chopped', 'sample pack', 'sample library', 'sample-based']
}

quality_indicators = {
    'polished': ['polished', 'pristine', 'professional', 'slick', 'hi-fi', 'clean production', 'refined', 'meticulous', 'well-crafted', 'immaculate', 'spotless'],
    'raw': ['raw', 'rough', 'unpolished', 'lo-fi', 'DIY', 'home recorded', 'bedroom', 'gritty', 'unfiltered', 'unrefined', 'earthy', 'untamed'],
    'experimental': ['experimental production', 'unconventional production', 'avant-garde production', 'innovative production', 'boundary-pushing', 'radical', 'cutting-edge'],
}

style_indicators = {
    'analog': ['analog', 'analogue', 'tape', 'vintage gear', 'warm', 'tube', 'retro', 'old-school', 'classic', 'magnetic'],
    'digital': ['digital', 'computer', 'DAW', 'plugin', 'modern production', 'software', 'virtual', 'electronic', 'high-tech', 'processed'],
    'minimalist': ['minimal production', 'sparse production', 'stripped down', 'barebones', 'simple', 'uncluttered', 'austere', 'streamlined'],
    'maximalist': ['wall of sound', 'dense production', 'layered production', 'lush production', 'opulent', 'extravagant', 'grand', 'ornate', 'complex']
}

mood_keywords = {
    'sad': ['sad', 'melancholy', 'somber', 'mournful', 'depressing', 'gloomy', 'sorrowful', 'dejected', 'despondent', 'forlorn', 'blue', 'downcast', 'heartbroken', 'tearful', 'woeful'],
    'happy': ['happy', 'joyful', 'cheerful', 'upbeat', 'celebratory', 'euphoric', 'elated', 'ecstatic', 'jubilant', 'exuberant', 'gleeful', 'content', 'delighted', 'sunny', 'radiant'],
    'angry': ['angry', 'aggressive', 'furious', 'hostile', 'violent', 'rage', 'wrathful', 'indignant', 'incensed', 'livid', 'irate', 'fuming', 'enraged', 'provoked', 'stormy'],
    'calm': ['calm', 'peaceful', 'serene', 'tranquil', 'soothing', 'gentle', 'placid', 'still', 'quiet', 'restful', 'composed', 'relaxed', 'unruffled', 'easygoing', 'mellow'],
    'anxious': ['anxious', 'tense', 'nervous', 'uneasy', 'restless', 'paranoid', 'edgy', 'jittery', 'worried', 'fraught', 'apprehensive', 'fretful', 'agitated', 'distressed', 'unsettled'],
    'romantic': ['romantic', 'intimate', 'sensual', 'loving', 'tender', 'affectionate', 'passionate', 'amorous', 'ardent', 'devoted', 'enamored', 'sweet', 'caring'],
    'dark': ['dark', 'sinister', 'ominous', 'foreboding', 'menacing', 'bleak', 'grim', 'shadowy', 'murky', 'eerie', 'gothic', 'macabre', 'creepy', 'dismal', 'spooky'],
    'dreamy': ['dreamy', 'ethereal', 'hazy', 'floating', 'otherworldly', 'surreal', 'hypnotic', 'trance-like', 'mystical', 'fantastical', 'illusory', 'imaginative', 'soft-focus', 'whimsical'],
    'melancholic': ['melancholic', 'wistful', 'pensive', 'reflective', 'contemplative', 'introspective', 'nostalgic', 'brooding', 'moody', 'regretful', 'yearning', 'sentimental'],
    'uplifting': ['uplifting', 'inspiring', 'hopeful', 'optimistic', 'encouraging', 'motivating', 'empowering', 'heartening', 'rejuvenating', 'buoyant', 'spirited', 'positive']
}

energy_indicators = {
    'high': ['energetic', 'explosive', 'frenetic', 'intense', 'powerful', 'driving', 'relentless', 'vigorous', 'dynamic', 'forceful', 'robust', 'spirited'],
    'medium': ['moderate', 'steady', 'consistent', 'balanced', 'even', 'regular', 'controlled', 'measured', 'stable'],
    'low': ['slow', 'subdued', 'mellow', 'laid-back', 'downtempo', 'languid', 'gentle', 'soft', 'calm', 'easygoing', 'peaceful']
}

polarizing_phrases = [
    'love it or hate it', 'divisive', 'polarizing', 'marmite',
    'not for everyone', 'challenging listen', 'acquired taste',
    'controversial', 'some will love', 'some will hate',
    'won\'t appeal to everyone', 'turn off some listeners',
     'split opinion', 'controversy', 'debated', 'hotly contested', 'mixed reactions', 'contentious', 'provokes discussion', 'sparks debate', 'divided audience'
]

novelty_positive = [
    'innovative', 'groundbreaking', 'fresh', 'original', 'unique',
    'pioneering', 'trailblazing', 'inventive', 'creative', 'novel',
    'revolutionary', 'game-changer', 'unprecedented',
     'trailblazing', 'state-of-the-art', 'forward-thinking', 'cutting-edge', 'next-level', 'breakthrough', 'trendsetting', 'visionary', 'inventive'
]

novelty_negative = [
    'derivative', 'formulaic', 'predictable', 'generic', 'clichéd',
    'unoriginal', 'by-the-numbers', 'paint-by-numbers', 'cookie-cutter',
    'rehash', 'retread', 'stale', 'tired',
     'played out', 'worn', 'repetitive', 'unimaginative', 'stagnant', 'cliché', 'hackneyed', 'old hat', 'overused'
]

context_keywords = {
    'party': ['party', 'club', 'dancefloor', 'dancing', 'celebration', 'festivity', 'rave', 'get-together', 'social', 'nightlife'],
    'study': ['focus', 'concentration', 'background', 'ambient', 'work', 'reading', 'studying', 'academic', 'library', 'quiet time'],
    'workout': ['energetic', 'pump', 'gym', 'exercise', 'adrenaline', 'training', 'fitness', 'cardio', 'strength', 'endurance'],
    'relaxation': ['relax', 'chill', 'unwind', 'calm', 'soothing', 'meditative', 'rest', 'peace', 'decompress', 'serenity'],
    'driving': ['driving', 'road trip', 'cruising', 'highway', 'commute', 'journey', 'travel', 'car ride', 'long drive', 'scenic route'],
    'intimate': ['intimate', 'bedroom', 'late night', 'romantic setting', 'private', 'close', 'personal', 'cozy', 'affectionate', 'tender moment'],
    'headphones': ['headphone', 'headphones', 'close listening', 'detailed listening', 'isolation', 'private listening', 'audiophile', 'immersive', 'focused listening', 'studio']
}

era_keywords = {
    'vintage': ['vintage', 'retro', 'throwback', 'nostalgic', 'classic sound', 'old-school', 'antique', 'heritage', 'historic', 'retrograde'],
    'contemporary': ['modern', 'contemporary', 'current', 'today', 'now', 'up-to-date', 'present-day', 'recent', 'new', 'latest'],
    'timeless': ['timeless', 'ageless', 'eternal', 'enduring', 'perpetual', 'unending', 'infinite', 'everlasting', 'classic']
}

lyrical_theme_keywords = {
    'love_romance': ['love', 'romance', 'relationship', 'heartbreak', 'longing', 'desire', 'affection', 'devotion', 'infatuation', 'crush', 'passion', 'yearning', 'attachment', 'adoration', 'fondness'],
    'social_political': ['political', 'protest', 'activism', 'justice', 'inequality', 'revolution', 'oppression', 'freedom', 'rights', 'resistance', 'civil rights', 'democracy', 'equality', 'campaign', 'movement'],
    'personal_growth': ['growth', 'self-discovery', 'identity', 'coming-of-age', 'maturity', 'transformation', 'evolution', 'awakening', 'progress', 'development', 'change', 'self-improvement', 'journey'],
    'existential': ['existential', 'mortality', 'death', 'meaning', 'purpose', 'existence', 'nihilism', 'void', 'absurdity', 'philosophy', 'reflection', 'searching', 'questioning', 'life'],
    'urban_life': ['city', 'urban', 'street', 'neighborhood', 'metropolitan', 'downtown', 'cityscape', 'concrete jungle', 'suburb', 'inner city', 'skyscraper', 'traffic', 'commute'],
    'nature': ['nature', 'landscape', 'environment', 'earth', 'wilderness', 'forest', 'ocean', 'mountains', 'sky', 'seasons', 'river', 'valley', 'meadow', 'flora', 'fauna'],
    'nostalgia_memory': ['memory', 'past', 'nostalgia', 'reminisce', 'childhood', 'youth', 'remembrance', 'bygone', 'yesterday', 'flashback', 'recollection', 'sentiment', 'old days', 'retro'],
    'struggle_hardship': ['struggle', 'hardship', 'pain', 'suffering', 'adversity', 'challenge', 'obstacle', 'burden', 'trial', 'difficulty', 'ordeal', 'tribulation', 'fight', 'battle'],
    'celebration': ['celebration', 'joy', 'party', 'triumph', 'victory', 'success', 'achievement', 'glory', 'festivity', 'jubilation', 'rejoicing', 'honor', 'commemoration', 'toast'],
    'spirituality': ['spiritual', 'faith', 'divine', 'transcendent', 'mystical', 'sacred', 'holy', 'prayer', 'meditation', 'enlightenment', 'worship', 'belief', 'soul', 'spirit', 'ritual'],
    'isolation': ['isolation', 'loneliness', 'solitude', 'alienation', 'disconnection', 'abandonment', 'forsaken', 'outcast', 'seclusion', 'withdrawal', 'apart', 'estrangement', 'distance'],
    'desire_lust': ['desire', 'lust', 'craving', 'yearning', 'hunger', 'temptation', 'seduction', 'passion', 'urge', 'want', 'need', 'greed', 'obsession'],
    'rebellion': ['rebellion', 'defiance', 'revolt', 'insurgency', 'nonconformity', 'dissent', 'mutiny', 'uprising', 'protest', 'resistance', 'insurrection', 'challenge', 'opposition']
}

general_theme_keywords = {
    'production_pristine': ['pristine', 'polished', 'clean', 'crisp', 'clarity', 'hi-fi', 'slick', 'refined', 'immaculate', 'spotless', 'well-produced'],
    'production_experimental': ['production', 'produced', 'sound design', 'sonic', 'mixing', 'mastering', 'layering', 'innovative', 'avant-garde', 'unconventional', 'creative'],
    'vocals_powerful': ['powerful vocals', 'soaring', 'belting', 'commanding voice', 'vocal prowess', 'dynamic', 'projecting', 'resonant', 'strong', 'robust'],
    'vocals_intimate': ['intimate vocals', 'whisper', 'breathy', 'tender voice', 'delicate singing', 'soft', 'gentle', 'close-mic', 'personal', 'subtle'],
    'vocals_harmonies': ['harmonies', 'vocal layers', 'backing vocals', 'choir', 'multitracked', 'choral', 'ensemble', 'group vocals', 'blended', 'unison'],
    'lyrics_poetic': ['poetic', 'wordplay', 'metaphor', 'lyrical', 'verse', 'poetry', 'imagery', 'symbolism', 'figurative', 'expressive'],
    'lyrics_narrative': ['storytelling', 'narrative', 'story', 'character', 'plot', 'tale', 'chronicle', 'account', 'fable', 'saga'],
    'emotion_melancholic': ['melancholy', 'sad', 'somber', 'mournful', 'wistful', 'sorrowful', 'blue', 'downcast', 'depressed', 'gloomy'],
    'emotion_joyful': ['joyful', 'happy', 'cheerful', 'euphoric', 'blissful', 'elated', 'gleeful', 'content', 'delighted', 'radiant'],
    'emotion_angry': ['angry', 'rage', 'furious', 'hostile', 'aggressive emotion', 'irate', 'fuming', 'stormy', 'provoked', 'enraged'],
    'emotion_vulnerable': ['vulnerable', 'fragile', 'exposed', 'heartbreak', 'raw emotion', 'open', 'unprotected', 'sensitive', 'tender', 'emotional'],
    'experimental_avant': ['experimental', 'avant-garde', 'unconventional', 'boundary-pushing', 'radical', 'cutting-edge', 'innovative', 'progressive', 'new'],
    'experimental_innovative': ['innovative', 'groundbreaking', 'pioneering', 'trailblazing', 'revolutionary', 'state-of-the-art', 'visionary', 'trendsetting', 'next-level'],
    'energy_explosive': ['explosive', 'relentless', 'frenetic', 'ferocious', 'breakneck', 'dynamic', 'vigorous', 'intense', 'forceful', 'robust'],
    'energy_driving': ['driving', 'propulsive', 'kinetic', 'momentum', 'pulsing', 'forward', 'thrust', 'energetic', 'moving', 'active'],
    'energy_chill': ['chill', 'relaxed', 'laid-back', 'mellow', 'easygoing', 'downtempo', 'soothing', 'calm', 'gentle', 'peaceful'],
    'atmosphere_dreamy': ['dreamy', 'ethereal', 'otherworldly', 'floating', 'weightless', 'soft-focus', 'whimsical', 'fantastical', 'illusory', 'mystical'],
    'atmosphere_dark': ['dark', 'brooding', 'ominous', 'menacing', 'sinister', 'foreboding', 'gothic', 'macabre', 'creepy', 'dismal'],
    'atmosphere_spacious': ['spacious', 'expansive', 'vast', 'open', 'airy', 'wide', 'roomy', 'unconfined', 'broad', 'large'],
    'atmosphere_claustrophobic': ['claustrophobic', 'suffocating', 'dense', 'oppressive', 'cramped', 'restricted', 'tight', 'closed-in', 'packed', 'overcrowded'],
    'atmosphere_cinematic': ['cinematic', 'soundtrack', 'film score', 'orchestral sweep', 'epic', 'grand', 'movie-like', 'dramatic', 'theatrical', 'score'],
    'rhythm_syncopated': ['syncopated', 'off-kilter', 'complex rhythm', 'polyrhythm', 'irregular', 'shifting', 'unpredictable', 'jazzy', 'groove', 'beat'],
    'rhythm_hypnotic': ['hypnotic', 'repetitive', 'trance', 'cyclical', 'looping', 'mesmerizing', 'entrancing', 'persistent', 'steady', 'drone'],
    'rhythm_groovy': ['groovy', 'funky', 'swagger', 'bounce', 'pocket', 'swing', 'catchy', 'infectious', 'danceable', 'rhythmic'],
    'texture_lush': ['lush', 'rich', 'thick', 'full', 'abundant', 'opulent', 'layered', 'dense', 'vivid', 'colorful'],
    'texture_sparse': ['sparse', 'minimal', 'bare', 'stripped', 'skeletal', 'thin', 'open', 'uncluttered', 'simple', 'austere'],
    'genre_fusion': ['fusion', 'blend', 'hybrid', 'cross-genre', 'mixing genres', 'combination', 'merging', 'integrated', 'mixed', 'eclectic'],
    'nostalgic_retro': ['nostalgic', 'throwback', 'retro', 'vintage', '70s', '80s', '90s', 'old-school', 'classic', 'heritage'],
    'nostalgic_timeless': ['timeless', 'classic', 'enduring', 'ageless', 'eternal', 'perpetual', 'unending', 'everlasting', 'infinite', 'legendary'],
    'danceable': ['danceable', 'club', 'dancefloor', 'party', 'move', 'groovy', 'upbeat', 'catchy', 'rhythmic', 'energetic'],
    'catchy': ['catchy', 'hook', 'memorable', 'infectious', 'earworm', 'anthemic', 'singalong', 'repetitive', 'sticky', 'addictive'],
    'technical_virtuosic': ['virtuosic', 'technical mastery', 'precision', 'skillful', 'dexterity', 'prowess', 'expert', 'accomplished', 'talented', 'gifted'],
    'technical_complex': ['complex', 'intricate', 'sophisticated', 'multifaceted', 'elaborate', 'ornate'],
    'technical_virtuosic': ['virtuosic', 'technical mastery', 'precision', 'skillful', 'dexterity', 'prowess', 'expert'],
    'technical_complex': ['complex', 'intricate', 'sophisticated', 'multifaceted', 'elaborate', 'ornate', 'detailed', 'advanced', 'complicated', 'layered'],
    'raw_gritty': ['gritty', 'rough', 'unpolished', 'lo-fi', 'DIY', 'rough-hewn', 'coarse', 'earthy', 'unrefined', 'raw'],
    'raw_visceral': ['visceral', 'primal', 'intense', 'immediate', 'gut-wrenching', 'powerful', 'raw', 'emotional', 'instinctive', 'forceful'],
    'uplifting': ['uplifting', 'hopeful', 'optimistic', 'triumphant', 'inspiring', 'elevating', 'soaring', 'heartening', 'rejuvenating', 'positive'],
    'introspective': ['introspective', 'reflective', 'contemplative', 'meditative', 'philosophical', 'thoughtful', 'self-aware', 'pensive', 'ruminative', 'inward'],
    'playful': ['playful', 'whimsical', 'quirky', 'tongue-in-cheek', 'irreverent', 'cheeky', 'mischievous', 'fun', 'lighthearted', 'jovial'],
    'sensual': ['sensual', 'sexy', 'sultry', 'seductive', 'erotic', 'steamy', 'provocative', 'passionate', 'intimate', 'alluring'],
    'psychedelic': ['psychedelic', 'trippy', 'mind-bending', 'hallucinogenic', 'kaleidoscopic', 'lysergic', 'surreal', 'otherworldly', 'fantastical', 'dreamlike'],
    'abrasive': ['abrasive', 'harsh', 'confrontational', 'challenging', 'jarring', 'grating', 'caustic', 'rough', 'discordant', 'unpleasant', 'strident'],
    'beautiful': ['beautiful', 'gorgeous', 'stunning', 'exquisite', 'lovely', 'sublime', 'magnificent', 'radiant', 'breathtaking', 'enchanting'],
    'chaotic': ['chaotic', 'frantic', 'messy', 'dissonant', 'turbulent', 'frenetic', 'wild', 'anarchic', 'disordered', 'unpredictable'],
    'cohesive': ['cohesive', 'unified', 'consistent', 'focused', 'tight', 'seamless', 'integrated', 'harmonious', 'well-structured', 'connected'],
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
    'ambient': ['ambient', 'environmental', 'soundscape', 'textural', 'immersive', 'enveloping', 'atmospheric', 'background', 'spacey', 'ethereal'],
    'confrontational': ['confrontational', 'provocative', 'challenging', 'unsettling', 'discomforting', 'abrasive', 'bold', 'assertive', 'defiant', 'controversial'],
    'euphoric': ['euphoric', 'ecstatic', 'blissful', 'transcendent', 'rapturous', 'exultant', 'elated', 'joyful', 'uplifted', 'overjoyed'],
    'haunting': ['haunting', 'ghostly', 'spectral', 'eerie', 'uncanny', 'unsettling', 'lingering', 'spooky', 'mysterious', 'chilling'],
    'majestic': ['majestic', 'grand', 'epic', 'monumental', 'towering', 'imposing', 'regal', 'noble', 'magnificent', 'stately'],
    'vulnerable': ['vulnerable', 'exposed', 'bare', 'unguarded', 'open', 'confessional', 'fragile', 'sensitive', 'raw', 'emotional']
}
