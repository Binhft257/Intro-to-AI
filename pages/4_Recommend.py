import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import os

# Layout UI
st.set_page_config(page_title="Vietnamese Songs by Emotion", page_icon="🎵", layout="wide")
st.sidebar.header("🎧 Vietnamese Song Suggestions")
st.header('🔍 Vietnamese Song Recommendations Based on Emotion')

# Load track data
@st.cache_data
def load_tracks(filename):
    return pd.read_csv(filename, lineterminator="\n")

tracks = load_tracks("./data/filteredtracks.csv")
tracks.columns = tracks.columns.str.strip()

# Đọc dữ liệu huấn luyện từ file TXT
@st.cache_resource
def load_training_data():
    X_train, y_train = [], []
    dataset_path = "./data/emotion_dataset.txt"
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split("|")
                if len(parts) == 2:
                    X_train.append(parts[0])
                    y_train.append(parts[1])
    else:
        st.error("Không tìm thấy file ./data/emotion_dataset.txt")
    return X_train, y_train

X_train, y_train = load_training_data()

# Tạo model Naive Bayes từ dữ liệu đã load
@st.cache_resource
def train_model(X, y):
    # Sử dụng TfidfVectorizer với ngram_range=(1,2)
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), MultinomialNB())
    model.fit(X, y)
    return model

classifier = train_model(X_train, y_train)

# Map cảm xúc → thể loại nhạc Việt
emotion_to_viet_genres = {
    "joy": ["vinahouse"],
    "sadness": ["vietnamese bolero", "bolero"],
    "anger": ["vietnamese hip hop"],
    "love": ["vietnamese lo-fi", "vietnamese bolero"],
    "fear": ["vietnamese ballad", "vietnamese acoustic"],
}

# Giao diện input
user_input = st.text_input("✍️ Enter a sentence expressing your emotion:")


if user_input:
    result = classifier.predict([user_input])[0]
    st.success(f"💡 Detected emotion: **{result.upper()}**")

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

    st.write("🎶 **Suggested song:**")

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
