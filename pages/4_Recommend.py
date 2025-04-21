import streamlit as st
import pandas as pd
import numpy as np
import ast
from transformers import pipeline

# Layout UI
st.set_page_config(page_title="Nhạc Việt theo cảm xúc", page_icon="🎵", layout="wide")
st.sidebar.header("🎧 Gợi ý Nhạc Việt")
st.header('🔍 Gợi ý bài hát tiếng Việt dựa theo cảm xúc')

# Load CSV
@st.cache_data
def load_tracks(filename):
    return pd.read_csv(filename, lineterminator="\n")

tracks = load_tracks("./data/filteredtracks.csv")
tracks.columns = tracks.columns.str.strip()


# Load model phân tích cảm xúc
@st.cache_resource
def load_model():
    return pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")

classifier = load_model()
candidate_labels = ["joy", "sadness", "anger", "fear", "love"]

# Map cảm xúc → thể loại nhạc Việt
emotion_to_viet_genres = {
    "joy": ["vinahouse",],
    "sadness": ["vietnamese bolero", "bolero"],
    "anger": ["vietnamese hip hop"],
    "fear": ["vietnam indie", "vietnamese lo-fi"],
    "love": ["vietnamese lo-fi", "vietnamese bolero"]
}

# Giao diện input
user_input = st.text_input("✍️ Nhập câu tiếng Việt thể hiện cảm xúc:")

if user_input:
    result = classifier(user_input, candidate_labels)["labels"][0]
    st.success(f"💡 Cảm xúc phát hiện: **{result.upper()}**")

    # Chuẩn hóa genre field
    def clean_genres(col):
        try:
            return [g.lower().strip() for g in ast.literal_eval(col)]
        except:
            return []

    tracks['genres'] = tracks['genres'].apply(clean_genres)

    # Lọc nhạc Việt theo cảm xúc
    def suggest_tracks(emotion, n=3):
        genres = emotion_to_viet_genres.get(emotion.lower(), [])
        db = tracks.explode("genres")
        filtered = db[db["genres"].isin(genres)]
        return filtered.sample(n=min(n, len(filtered)))

    suggestions = suggest_tracks(result)

    st.write("🎶 **Gợi ý bài hát:**")
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]

    for i in range(min(3, len(suggestions))):
        uri = suggestions.iloc[i]["uri"]
        if uri.startswith("spotify:track:"):
            uri = uri.replace("spotify:track:", "")
        cols[i].markdown(
            f"<iframe src='https://open.spotify.com/embed/track/{uri}' width='100%' height='200' frameborder='0' allowtransparency='true' allow='encrypted-media'></iframe>",
            unsafe_allow_html=True
        )
