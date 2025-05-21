import streamlit as st
import pandas as pd
import numpy as np
import ast
import os
import cv2
import pathlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# App Configuration
st.set_page_config(page_title="Vietnamese Songs by Emotion", page_icon="🎵", layout="wide")
st.sidebar.header("🎧 Vietnamese Song Suggestions")
st.header('🎵 Vietnamese Song Recommendations Based on Emotion')

# Load tracks data
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent 
DATA_FILE = BASE_DIR / "data" / "file_chuan_4cot.csv"

@st.cache_data
def load_tracks():
    df = pd.read_csv(DATA_FILE, lineterminator="\n")

    df.columns = df.columns.str.strip()

    # Parse genres
    df['genres'] = df['genres'].apply(
        lambda x: [g.lower().strip() for g in ast.literal_eval(x)] 
                  if pd.notna(x) else []
    )
    return df

tracks = load_tracks()
# Load training data for text classifier
@st.cache_resource
def load_training_data():
    X_train, y_train = [], []
    path = "./data/emotion_dataset.txt"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split("|")
                if len(parts) == 2:
                    X_train.append(parts[0])
                    y_train.append(parts[1])
    return X_train, y_train

X_train, y_train = load_training_data()

# Train the text classifier
@st.cache_resource
def train_model(X, y):
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), MultinomialNB())
    model.fit(X, y)
    return model

classifier = train_model(X_train, y_train)

# Mapping emotion to Vietnamese genres
emotion_to_viet_genres = {
    "love": ["love"],
    "joy": ["vinahouse","rap","edm"],
    "sadness": ["lofi", "bolero","ballad"],
    "anger": ["rock"],
    "fear": ["indie"],
}

# Load face model and Haar Cascade
face_model = load_model("./data/video_based/finall_emotion_model.h5")
face_cascade = cv2.CascadeClassifier("./data/video_based/haarcascade_frontalface_default.xml")

# Emotion labels from model training
label_index_to_name = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

mapped_labels = label_index_to_name

# Tabs
tab1, tab2 = st.tabs(["📝 Text-Based", "📸 Webcam Capture-Based"])

# Tab 1 - Text 
with tab1:
    st.subheader("📝 Detect Emotion from Text")
    user_input = st.text_input("✍️ Enter a sentence expressing your emotion:")

    if user_input:
        detected_emotion = classifier.predict([user_input])[0]
        if detected_emotion in emotion_to_viet_genres:
            st.success(f"💡 Detected Emotion: **{detected_emotion.upper()}**")
            genres = emotion_to_viet_genres[detected_emotion]
            filtered = tracks.explode("genres")
            suggestions = filtered[filtered["genres"].isin(genres)].sample(n=min(3, len(filtered)))

            if not suggestions.empty:
                st.write("🎶 **Suggested Songs:**")
                cols = st.columns(3)
                for i in range(len(suggestions)):
                    uri = suggestions.iloc[i]["uri"].replace("spotify:track:", "")
                    cols[i].markdown(
                        f"<iframe src='https://open.spotify.com/embed/track/{uri}' width='100%' height='200' frameborder='0' allowtransparency='true' allow='encrypted-media'></iframe>",
                        unsafe_allow_html=True
                    )
            else:
                st.warning("No song suggestions found for this emotion.")
        else:
            st.warning("Detected emotion is not in the supported list (love, joy, sadness, anger, fear).")

# Tab 2 - Webcam Capture Based
with tab2:
    mapped_labels = {
    'angry': 'anger',
    'disgust': 'anger',
    'fear': 'fear',
    'happy': 'joy',
    'sad': 'sadness',
    'surprise': 'joy',
    'neutral': 'sadness'
}

    st.subheader("📸 Detect Emotion from Webcam or Image Upload")

    option = st.radio("Choose Input Method:", ["Webcam Capture", "Upload from File"])

    if option == "Webcam Capture":
        st.subheader("📸 Capture and Detect Emotion from Webcam")

        class VideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.frame = None
            def recv(self, frame):
                self.frame = frame.to_ndarray(format="bgr24")
                return frame

        ctx = webrtc_streamer(
            key="emotion-detection",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
        )

        if ctx.video_processor:
            st.info("Press the button below to capture an image from the webcam.")
            if st.button("📸 Capture Photo"):
                frame = ctx.video_processor.frame
                if frame is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray)

                    if len(faces) == 0:
                        st.warning("No face detected.")
                    else:
                        for (x, y, w, h) in faces:
                            face = gray[y:y + h, x:x + w]
                            face = cv2.resize(face, (48, 48))
                            face = face.astype("float32") / 255.0
                            face = np.expand_dims(face, axis=-1)
                            face = np.expand_dims(face, axis=0)

                            preds = face_model.predict(face)[0]
                            predicted_index = np.argmax(preds)
                            raw_label = label_index_to_name[predicted_index]
                            detected_emotion = mapped_labels.get(raw_label, None)


                            st.success(f"🖼️ Detected Emotion: **{raw_label.upper()}** (Class {predicted_index})")

                            if detected_emotion in emotion_to_viet_genres:
                                genres = emotion_to_viet_genres[detected_emotion]
                                filtered = tracks.explode("genres")
                                suggestions = filtered[filtered["genres"].isin(genres)].sample(n=min(3, len(filtered)))

                                if not suggestions.empty:
                                    st.write("🎶 **Suggested Songs:**")
                                    cols = st.columns(3)
                                    for i in range(len(suggestions)):
                                        uri = suggestions.iloc[i]["uri"].replace("spotify:track:", "")
                                        cols[i].markdown(
                                            f"<iframe src='https://open.spotify.com/embed/track/{uri}' width='100%' height='200' frameborder='0' allowtransparency='true' allow='encrypted-media'></iframe>",
                                            unsafe_allow_html=True
                                        )
                                else:
                                    st.warning("No song suggestions found for this emotion.")
                            else:
                                st.warning("No genre mapping available for this emotion.")
                            break
                else:
                    st.warning("No frame captured. Please ensure the webcam is working.")
    
    elif option == "Upload from File":
        st.subheader("📤 Upload an Image to Detect Emotion")
        uploaded_file = st.file_uploader("Upload an image (face visible)", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray)

            if len(faces) == 0:
                st.warning("No face detected in the uploaded image.")
            else:
                for (x, y, w, h) in faces:
                    face = gray[y:y + h, x:x + w]
                    face = cv2.resize(face, (48, 48))
                    face = face.astype("float32") / 255.0
                    face = np.expand_dims(face, axis=-1)
                    face = np.expand_dims(face, axis=0)

                    preds = face_model.predict(face)[0]
                    predicted_index = np.argmax(preds)
                    raw_label = label_index_to_name[predicted_index]
                    detected_emotion = mapped_labels.get(raw_label, None)


                    st.success(f"🖼️ Detected Emotion: **{raw_label.upper()}** (Class {predicted_index})")

                    if detected_emotion in emotion_to_viet_genres:
                        genres = emotion_to_viet_genres[detected_emotion]
                        filtered = tracks.explode("genres")
                        suggestions = filtered[filtered["genres"].isin(genres)].sample(n=min(3, len(filtered)))

                        if not suggestions.empty:
                            st.write("🎶 **Suggested Songs:**")
                            cols = st.columns(3)
                            for i in range(len(suggestions)):
                                uri = suggestions.iloc[i]["uri"].replace("spotify:track:", "")
                                cols[i].markdown(
                                    f"<iframe src='https://open.spotify.com/embed/track/{uri}' width='100%' height='200' frameborder='0' allowtransparency='true' allow='encrypted-media'></iframe>",
                                    unsafe_allow_html=True
                                )
                        else:
                            st.warning("No song suggestions found for this emotion.")
                    else:
                        st.warning("No genre mapping available for this emotion.")
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB")
