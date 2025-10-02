# app.py for Video Game Reviews

import streamlit as st
import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Video Game Review Analyzer",
    page_icon="🎮",
    layout="wide"
)

# ===================================================================
# !! MODIFIED SECTION !!
# This function now downloads the NLTK data silently without showing messages.
def download_nltk_data():
    packages = ['stopwords', 'wordnet', 'omw-1.4']
    for package in packages:
        try:
            nltk.data.find(f'corpora/{package}')
        except LookupError:
            # The download will happen quietly in the background.
            nltk.download(package, quiet=True)

# Run the silent download function at the start of the app
download_nltk_data()
# ===================================================================


# --- Asset Loading ---
@st.cache_resource
def load_models():
    """Loads all models and vectorizers for the video game dataset."""
    with open('models/sentiment_model_vg.pkl', 'rb') as f:
        sentiment_model = pickle.load(f)
    with open('models/tfidf_vectorizer_vg.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open('models/lda_model_vg.pkl', 'rb') as f:
        lda_model = pickle.load(f)
    with open('models/count_vectorizer_vg.pkl', 'rb') as f:
        count_vectorizer = pickle.load(f)
    return sentiment_model, tfidf_vectorizer, lda_model, count_vectorizer

# --- Helper Functions ---
def preprocess_text(text):
    if not isinstance(text, str): return ""
    stop_words = set(stopwords.words('english'))
    negation_words = {'not', 'no', 'nor'}
    stop_words = stop_words - negation_words
    text = re.sub(r"[^a-zA-Z]", " ", text).lower()
    tokens = text.split()
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(clean_tokens)

ASPECT_KEYWORDS = {
    "gameplay": ["gameplay", "fun", "play", "mechanics", "controls", "boring"],
    "graphics": ["graphics", "visuals", "art", "style", "scenery", "beautiful", "look"],
    "story": ["story", "narrative", "plot", "characters", "ending", "writing"],
    "performance": ["bugs", "glitches", "crash", "performance", "lag", "fps", "error"]
}

def get_aspect_sentiments(review, model, vectorizer):
    clean_review = preprocess_text(review)
    review_tokens = clean_review.split()
    aspect_sentiments = {}
    
    for aspect, keywords in ASPECT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in review_tokens:
                try:
                    keyword_index = review_tokens.index(keyword)
                    start = max(0, keyword_index - 10)
                    end = min(len(review_tokens), keyword_index + 11)
                    context_window = " ".join(review_tokens[start:end])
                    vectorized_window = vectorizer.transform([context_window])
                    prediction = model.predict(vectorized_window)[0]
                    aspect_sentiments[aspect] = prediction
                    break 
                except:
                    pass
    return aspect_sentiments

# --- Load all assets ---
try:
    sentiment_model, tfidf_vectorizer, lda_model, count_vectorizer = load_models()
except FileNotFoundError:
    st.error("Model files not found! Please run the training notebook to create and save the model files in a 'models' folder.")
    st.stop()

# --- App Layout ---
st.title("🎮 Video Game Review Analyzer")
st.markdown("A tool to analyze sentiment, aspects, and topics from video game reviews, based on the client's dataset.")

# Sidebar for navigation
st.sidebar.title("Analysis Tools")
app_mode = st.sidebar.selectbox(
    "Choose an analysis tool:",
    ["Sentiment Predictor", "Deeper Analysis (ABSA)", "Topic Explorer"]
)

# --- Page 1: Sentiment Predictor ---
if app_mode == "Sentiment Predictor":
    st.header("Overall Sentiment Prediction")
    st.markdown("Enter a game review to predict its sentiment (Positive, Neutral, or Negative).")
    
    user_input = st.text_area("Review Text:", "This is a solid game, but it has too many bugs.", height=150)
    
    if st.button("Predict Sentiment"):
        if user_input:
            clean_input = preprocess_text(user_input)
            vectorized_input = tfidf_vectorizer.transform([clean_input])
            prediction = sentiment_model.predict(vectorized_input)[0]
            
            if prediction == "positive":
                st.success(f"Predicted Sentiment: **Positive** 👍")
            elif prediction == "neutral":
                st.warning(f"Predicted Sentiment: **Neutral** 😐")
            else:
                st.error(f"Predicted Sentiment: **Negative** 👎")
        else:
            st.warning("Please enter some text to analyze.")

# --- Page 2: Deeper Analysis (ABSA) ---
elif app_mode == "Deeper Analysis (ABSA)":
    st.header("Aspect-Based Sentiment Analysis")
    st.markdown("Find the sentiment for specific aspects like **gameplay**, **graphics**, **story**, and **performance**.")
    
    user_input = st.text_area("Review Text:", "The gameplay is really fun and the story is great, but the performance is terrible with lots of bugs.", height=150)
    
    if st.button("Analyze Aspects"):
        if user_input:
            aspects = get_aspect_sentiments(user_input, sentiment_model, tfidf_vectorizer)
            if not aspects:
                st.info("No specific aspects were mentioned in this review.")
            else:
                st.write("### Aspect Sentiment Results:")
                for aspect, sentiment in aspects.items():
                    if sentiment == "positive":
                        st.markdown(f"**{aspect.title()}:** <span style='color:green;'>**Positive**</span>", unsafe_allow_html=True)
                    elif sentiment == "neutral":
                        st.markdown(f"**{aspect.title()}:** <span style='color:orange;'>**Neutral**</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**{aspect.title()}:** <span style='color:red;'>**Negative**</span>", unsafe_allow_html=True)
        else:
            st.warning("Please enter a review to analyze.")

# --- Page 3: Topic Explorer ---
elif app_mode == "Topic Explorer":
    st.header("Discover Hidden Topics in Reviews")
    st.markdown("These topics were automatically discovered from the video game reviews dataset using the LDA algorithm.")

    feature_names = count_vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda_model.components_):
        st.subheader(f"Topic #{topic_idx + 1}")
        topic_words = " | ".join([feature_names[i] for i in topic.argsort()[:-10 - 1:-1]])
        st.markdown(f"**Keywords:** `{topic_words}`")