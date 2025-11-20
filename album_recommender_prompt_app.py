import os
import streamlit as st
from album_recommender_model import EnhancedRecommender
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import logging
import sys
import random

os.environ['PYTHONWARNINGS'] = 'ignore'

# Suppress all warnings at multiple levels
warnings.filterwarnings('ignore')
logging.getLogger('streamlit').setLevel(logging.CRITICAL)
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(
    logging.CRITICAL)


class SuppressStderr:
    def write(self, msg):
        if 'ScriptRunContext' not in msg:
            sys.__stderr__.write(msg)

    def flush(self):
        pass


sys.stderr = SuppressStderr()

st.set_page_config(page_title="Album Recommender", page_icon="🎵")

with open("app_styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def get_album_art(url, _retry=0):
    """Fetch album art from Pitchfork review page"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=3)

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.content, 'html.parser')

        # Look for responsive image container
        responsive_container = soup.find(
            'div', class_=lambda x: x and 'responsiveimagecontainer' in x.lower())
        if responsive_container:
            img_tag = responsive_container.find('img')
            if img_tag:
                # Try srcset first (largest version)
                if img_tag.get('srcset'):
                    srcset = img_tag.get('srcset')
                    images = [s.strip().split(' ')[0]
                              for s in srcset.split(',')]
                    if images:
                        return images[-1]  # Last one is usually largest
                # Fallback to src
                if img_tag.get('src'):
                    return img_tag['src']

        # Fallback to og:image
        og_image = soup.find('meta', property='og:image')
        if og_image and og_image.get('content'):
            return og_image['content']

        # Try twitter:image as fallback
        twitter_image = soup.find('meta', attrs={'name': 'twitter:image'})
        if twitter_image and twitter_image.get('content'):
            return twitter_image['content']

        return None
    except Exception as e:
        return None


def get_album_arts_parallel(urls):
    """Fetch multiple album arts in parallel"""
    album_arts = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(
            get_album_art, url): url for url in urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                album_arts[url] = future.result()
            except:
                album_arts[url] = None
    return album_arts


@st.cache_resource
def load_recommender():
    print("Loading Album Recommendation Models: TF-IDF (text similarity), semantic embeddings (contextual meaning), and feature-engineered vectors (album attributes) for fast, accurate album recommendations.")
    recommender = EnhancedRecommender()
    # Try loading pre-built models first
    if not recommender.load_models():
        print("Pre-built models not found. Building models from scratch. This may take a moment...")
        recommender.build_models()
    return recommender


st.title("Album Recommendation System")

# Add spacing after title
st.markdown("<br>", unsafe_allow_html=True)

# Random placeholder examples
placeholder_examples = [
    "atmospheric electronic",
    "high-energy rock for workouts",
    "calm jazz for studying",
    "melancholic indie folk",
    "upbeat dance music for parties",
    "dark ambient soundscapes",
    "soulful R&B with smooth vocals",
    "experimental hip-hop",
    "dreamy shoegaze guitars",
    "aggressive punk energy",
    "romantic classical piano",
    "funky bass-driven grooves",
    "sad songs for rainy days",
    "uplifting anthems",
    "psychedelic rock jams",
    "nostalgic 80s synth-pop",
    "intense metal with heavy riffs",
    "chill lo-fi beats",
    "emotional singer-songwriter",
    "groovy disco vibes",
    "introspective bedroom pop",
    "epic orchestral soundtracks",
    "raw garage rock",
    "smooth bossa nova",
    "energetic ska-punk",
    "meditative new age",
    "gritty blues with soul",
    "futuristic synthwave",
    "powerful protest songs",
    "trippy psych-folk",
    "minimal techno for focus",
    "explosive noise rock",
    "jazzy hip-hop instrumentals",
    "haunting gothic post-punk",
    "feel-good summer pop",
    "cinematic post-rock",
    "intimate acoustic ballads",
    "funk-infused rock",
    "ethereal vocal harmonies",
    "driving country rock",
    "catchy power pop",
    "spiritual gospel choirs",
    "dynamic progressive metal",
    "laid-back reggae rhythms",
    "complex math rock",
    "soul-stirring blues rock",
    "hypnotic trance music",
    "raw lo-fi indie rock",
    "uplifting worship music",
    "cinematic electronic",
    "intense hardcore punk",
    "melodic death metal",
    "smooth jazz fusion",
    "ambient drone music",
    "funky soul classics",
    "darkwave synth-pop",
]

# Initialize random placeholder in session state
if 'placeholder_text' not in st.session_state:
    st.session_state.placeholder_text = random.choice(placeholder_examples)



user_prompt = st.text_input(
    "What kind of music are you looking for?",
    placeholder=f"e.g., '{st.session_state.placeholder_text}'",
    key="main_input"
)

# Three buttons: Clear, Recommend, Surprise Me
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    clear_button = st.button("🔄 Clear", use_container_width=True)
with col2:
    recommend_button = st.button(
        "🎵 Recommend", type="primary", use_container_width=True)
with col3:
    surprise_button = st.button("🎲 Surprise Me", use_container_width=True)


# Handle Clear button
if clear_button:
    # Clear all session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Handle Surprise Me button
if surprise_button:
    # Pick a random prompt from our examples
    surprise_prompt = random.choice(placeholder_examples)
    # Set it as the user prompt by storing in session state
    st.session_state.surprise_prompt = surprise_prompt
    user_prompt = surprise_prompt

# Use surprise prompt if it exists
if 'surprise_prompt' in st.session_state and not user_prompt:
    user_prompt = st.session_state.surprise_prompt
    # Clear it after use
    del st.session_state.surprise_prompt

# Trigger on button click OR when user presses Enter in text field
if recommend_button or surprise_button or user_prompt:
    if user_prompt:
        # Initialize session state for number of results to show
        if 'num_results' not in st.session_state:
            st.session_state.num_results = 5

        # Set default filter values if not defined
        sentiment_filter = None
        min_score = None
        theme_filter = None
        mood_filter = None
        context_filter = None
        novelty_filter = None
        artist_filter = None

        # Load recommender on demand (only when needed) and cache it
        if 'recommender' not in st.session_state:
            with st.spinner("Loading Recommendations..."):
                st.session_state.recommender = load_recommender()
        with st.spinner("Loading Recommendations..."):
            recommendations = st.session_state.recommender.recommend_diverse(
                user_prompt,
                top_n=st.session_state.num_results,
                min_score=min_score
            )
            st.session_state.all_recommendations = recommendations
            st.session_state.last_prompt = user_prompt

            if recommendations:
                # Fetch album arts for current batch
                urls_to_fetch = [rec['url']
                                 for rec in recommendations[:st.session_state.num_results]]
                album_arts = get_album_arts_parallel(urls_to_fetch)

                for i, rec in enumerate(recommendations[:st.session_state.num_results], 1):
                    # Add fade-in animation to newly loaded albums
                    fade_class = "fade-in" if i > (
                        st.session_state.num_results - 5) else ""
                    st.markdown(
                        f"<div class='{fade_class}'>", unsafe_allow_html=True)

                    # Display album art centered if available
                    album_art_url = album_arts.get(rec['url'])
                    if album_art_url:
                        st.markdown(
                            f"<div style='text-align: center; margin-bottom: 1rem;'><img src='{album_art_url}' style='border-radius: 8px;' /></div>", unsafe_allow_html=True)

                    st.markdown(
                        f"<h3 style='text-align: center;'>{rec['album']}</h3>", unsafe_allow_html=True)
                    st.markdown(
                        f"<p style='text-align: center;'><strong>Artist:</strong> {rec['artist']}</p>", unsafe_allow_html=True)
                    st.markdown(
                        f"<p style='text-align: center;'><strong>Genre:</strong> {rec['genre']} | <strong>Score:</strong> {rec['score']}/10 | <strong>Year:</strong> {int(rec['year'])}</p>", unsafe_allow_html=True)

                    # Display themes
                    if rec.get('themes') and rec['themes'] != 'N/A':
                        themes_list = [t.strip()
                                       for t in rec['themes'].split(',')]
                        cleaned_themes = []
                        for theme in themes_list:
                            if '_' in theme:
                                parts = theme.split('_', 1)
                                cleaned = parts[1].replace('_', ' ').title() if len(
                                    parts) > 1 else theme.replace('_', ' ').title()
                            else:
                                cleaned = theme.replace('_', ' ').title()
                            cleaned_themes.append(cleaned)
                        themes_display = ' • '.join(cleaned_themes)
                        st.markdown(
                            f"<p style='text-align: center; margin-bottom: 0;'><strong>Themes:</strong></p>", unsafe_allow_html=True)
                        st.markdown(
                            f"<p style='text-align: center; margin-top: 0;'>{themes_display}</p>", unsafe_allow_html=True)

                    # Collapsible sections for additional details
                    col1, col2 = st.columns(2)

                    with col1:
                        if rec['highlights'] and isinstance(rec['highlights'], str):
                            with st.expander("📝 Review Highlights"):
                                for highlight in rec['highlights'].split(' | ')[:3]:
                                    if highlight:
                                        st.markdown(f"- {highlight}")

                        if rec.get('instrumentation'):
                            with st.expander("🎸 Instrumentation"):
                                st.markdown(rec['instrumentation'])

                    with col2:
                        if rec.get('mood_energy'):
                            with st.expander("🎭 Mood & Energy"):
                                st.markdown(rec['mood_energy'])

                        if rec.get('listening_contexts'):
                            with st.expander("🎧 Best For"):
                                st.markdown(rec['listening_contexts'])

                    # Center the read full review link
                    st.markdown(
                        f"<div style='text-align: center; margin-top: 1rem;'><a href='{rec['url']}' target='_blank'>📖 Read Full Review</a></div>", unsafe_allow_html=True)
                    st.markdown("---")
                    st.markdown("</div>", unsafe_allow_html=True)

                # Show "Load More" button if not at maximum and we have more results
                if st.session_state.num_results < len(recommendations) and st.session_state.num_results < 20:
                    col1, col2, col3 = st.columns([2, 1, 2])
                    with col2:
                        if st.button("Load 5 More Albums", key="load_more", use_container_width=True):
                            st.session_state.num_results += 5
                            st.rerun()
            else:
                st.warning("No albums found. Try a different description.")
    else:
        st.info(
            "Tell us what you're in the mood for and we'll find the perfect albums!")

# Handle "Load More" when no new prompt (user just clicked the button)
if 'all_recommendations' in st.session_state and 'last_prompt' in st.session_state:
    recommendations = st.session_state.all_recommendations
    if recommendations and not (recommend_button or user_prompt):
        # Fetch album arts for current batch
        urls_to_fetch = [rec['url']
                         for rec in recommendations[:st.session_state.num_results]]
        album_arts = get_album_arts_parallel(urls_to_fetch)

        # Display cached recommendations
        for i, rec in enumerate(recommendations[:st.session_state.num_results], 1):
            # Add fade-in animation to newly loaded albums
            fade_class = "fade-in" if i > (
                st.session_state.num_results - 5) else ""
            st.markdown(f"<div class='{fade_class}'>", unsafe_allow_html=True)

            # Display album art centered if available
            album_art_url = album_arts.get(rec['url'])
            if album_art_url:
                st.markdown(
                    f"<div style='text-align: center; margin-bottom: 1rem;'><img src='{album_art_url}' style='border-radius: 8px;' /></div>", unsafe_allow_html=True)

            st.markdown(
                f"<h3 style='text-align: center;'>{rec['album']}</h3>", unsafe_allow_html=True)
            st.markdown(
                f"<p style='text-align: center;'><strong>Artist:</strong> {rec['artist']}</p>", unsafe_allow_html=True)
            st.markdown(
                f"<p style='text-align: center;'><strong>Genre:</strong> {rec['genre']} | <strong>Score:</strong> {rec['score']}/10 | <strong>Year:</strong> {int(rec['year'])}</p>", unsafe_allow_html=True)

            # Display themes
            if rec.get('themes') and rec['themes'] != 'N/A':
                themes_list = [t.strip() for t in rec['themes'].split(',')]
                cleaned_themes = []
                for theme in themes_list:
                    if '_' in theme:
                        parts = theme.split('_', 1)
                        cleaned = parts[1].replace('_', ' ').title() if len(
                            parts) > 1 else theme.replace('_', ' ').title()
                    else:
                        cleaned = theme.replace('_', ' ').title()
                    cleaned_themes.append(cleaned)
                themes_display = ' • '.join(cleaned_themes)
                st.markdown(
                    f"<p style='text-align: center; margin-bottom: 0;'><strong>Themes:</strong></p>", unsafe_allow_html=True)
                st.markdown(
                    f"<p style='text-align: center; margin-top: 0;'>{themes_display}</p>", unsafe_allow_html=True)

            # Collapsible sections for additional details
            col1, col2 = st.columns(2)

            with col1:
                if rec['highlights'] and isinstance(rec['highlights'], str):
                    with st.expander("📝 Review Highlights"):
                        for highlight in rec['highlights'].split(' | ')[:3]:
                            if highlight:
                                st.markdown(f"- {highlight}")

                if rec.get('instrumentation'):
                    with st.expander("🎸 Instrumentation"):
                        st.markdown(rec['instrumentation'])

            with col2:
                if rec.get('mood_energy'):
                    with st.expander("🎭 Mood & Energy"):
                        st.markdown(rec['mood_energy'])

                if rec.get('listening_contexts'):
                    with st.expander("🎧 Best For"):
                        st.markdown(rec['listening_contexts'])

            # Center the read full review link
            st.markdown(
                f"<div style='text-align: center; margin-top: 1rem;'><a href='{rec['url']}' target='_blank'>📖 Read Full Review</a></div>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("</div>", unsafe_allow_html=True)

        # Show "Load More" button if not at maximum and we have more results
        if st.session_state.num_results < len(recommendations) and st.session_state.num_results < 20:
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                if st.button("Load 5 More Albums", key="load_more_cached", use_container_width=True):
                    st.session_state.num_results += 5
                    st.rerun()
