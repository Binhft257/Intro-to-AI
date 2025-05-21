Emotion-Based Music Recommendation System

This project is a web application that recommends Vietnamese songs based on the user's emotional state, detected through either text input or facial expression analysis via webcam or uploaded images. It leverages machine learning models, including Naive Bayes for text-based emotion detection and a convolutional neural network (CNN) for facial emotion recognition, to suggest songs from a Spotify dataset tailored to the detected emotions.
Features

Text-Based Emotion Detection: Users can input a Vietnamese sentence, and the system uses a Naive Bayes classifier to detect emotions (joy, sadness, anger, fear, love) and recommend songs.
Facial Emotion Recognition: Users can capture an image via webcam or upload a photo, and a pre-trained CNN model detects emotions (angry, disgust, fear, happy, neutral, sad, surprise) to suggest songs.
Spotify Integration: Songs are sourced from Spotify playlists, crawled using the Spotify API, and matched to emotions via predefined genre mappings.
Interactive UI: Built with Streamlit, the app offers a user-friendly interface with tabs for text and image-based recommendations.
Responsive Design: Incorporates Bootstrap for styling and supports embedded Spotify players for song previews.

Project Structure

main.py: The main Streamlit app file, defining the homepage with project details and links to external resources (GitHub, Colab, etc.).
crawlData.py: Script to crawl Spotify playlists using the Spotify API and save song data (track ID, name, URI, genre) to a CSV file.
4_Recommend.py: The recommendation module, handling emotion detection (text and image-based) and song suggestions.
data/: Directory containing:
file_chuan_4cot.csv: Dataset of songs with track ID, name, URI, and genres.
emotion_dataset.txt: Training data for the text-based emotion classifier.
video_based/finall_emotion_model.h5: Pre-trained CNN model for facial emotion recognition.
video_based/haarcascade_frontalface_default.xml: Haar Cascade for face detection.


icons/: Directory with images for the UI (cover, Streamlit, Spotify, Colab logos).

Prerequisites

Python 3.8+
Spotify Developer Account (for API credentials)
Webcam (for facial emotion detection)

Installation

Clone the Repository:
git clone https://github.com/Binhft257/Intro-to-AI
cd Emotion-Based-Music-Recommendation-System


Install Dependencies:Install the required Python packages using:
pip install -r requirements.txt


Set Up Spotify API Credentials:

Create a Spotify Developer account and set up an app at Spotify Developer Dashboard.
Obtain your client_id and client_secret.
Update crawlData.py with your credentials:client_id = 'your_client_id',
client_secret = 'your_client_secret'




Prepare the Dataset:

Run crawlData.py to fetch Spotify playlist data:python crawlData.py

This generates multiple CSV files, each corresponding to a different music genre (e.g., pop.csv, rap.csv, rock.csv, etc.). Merge or process it into data/file_chuan_4cot.csv with columns: track_id, name, uri, genres.
Ensure data/emotion_dataset.txt contains training data in the format: text|emotion.


Download Pre-trained Model:

Ensure data/video_based/finall_emotion_model.h5 and data/video_based/haarcascade_frontalface_default.xml are in place. 



Usage

Run the Streamlit App:
streamlit run main.py

This starts the app at http://localhost:8501.

Text-Based Recommendations:

Navigate to the "Text-Based" tab.
Enter a Vietnamese sentence expressing an emotion (e.g., "Tôi rất vui hôm nay").
The app detects the emotion and suggests up to three Spotify songs matching the emotion's genre.


Webcam or Image-Based Recommendations:

Navigate to the "Webcam Capture-Based" tab.
Choose "Webcam Capture" to use your webcam or "Upload from File" to upload an image.
For webcam: Click "Capture Photo" to analyze your facial expression.
For upload: Select a JPG/PNG image with a visible face.
The app detects the emotion and suggests songs based on the mapped genre.



Emotion-to-Genre Mapping
The system maps detected emotions to Vietnamese music genres:

Love: Love songs
Joy: Vinahouse, Rap, EDM
Sadness: Lo-fi, Bolero, Ballad
Anger: Rock
Fear: Indie

For facial recognition, emotions are mapped as follows:

Angry, Disgust → Anger
Fear → Fear
Happy, Surprise → Joy
Sad, Neutral → Sadness

Limitations

The text classifier is trained on a simple dataset and may struggle with complex or ambiguous Vietnamese text.
Facial emotion detection requires clear, well-lit images and may not work well with multiple faces or poor lighting.
Song recommendations are limited to the genres and tracks in file_chuan_4cot.csv.
The Spotify API requires an active internet connection and valid credentials.

Future Improvements

Enhance the text classifier with a larger, more diverse dataset and advanced NLP models.
Improve facial emotion recognition with a custom-trained model for Vietnamese users.
Expand the song dataset to include more genres and artists.
Add support for real-time emotion detection via continuous webcam streaming.


