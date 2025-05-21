import base64
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Home",
    page_icon=":earth_africa:",
)

def load_bootstrap():
    return st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded
with st.container():
    st.title("Emotion Based Music Recommendation System")
    for i in range(2):
        st.markdown('#')
    st.caption('This project recommends Vietnamese songs based on the emotion expressed in the text entered by the user in Vietnamese.')
    st.caption('The model is trained using Naive Bayes with a simple Vietnamese emotion dataset containing four labels: joy, sadness, anger, fear, and love.')
    st.caption('It helps users quickly and easily discover Vietnamese music that matches their mood.')

    for i in range(2):
        st.markdown('#')
    st.markdown('#####')
    st.markdown('---')

    col1, col2, col3 = st.columns((1.4,0.85,0.85), gap='large')
    with col1:
        st.empty()
        st.empty()
        st.markdown("<a href='https://docs.streamlit.io/library/get-started'><img src='data:image/png;base64,{}' class='img-fluid' width=80%/></a>".format(img_to_bytes('./icons/streamlit.png')), unsafe_allow_html=True)
    with col2:
        st.markdown("<a href='https://developer.spotify.com/'><img src='data:image/png;base64,{}' class='img-fluid' width=40%/></a>".format(img_to_bytes('./icons/spotify.png')), unsafe_allow_html=True)
    with col3:
        st.markdown("<a href='https://colab.research.google.com/drive/1ahxyp8i9Ngy2nyA5THSOwDzVS99prLMF?usp=sharing'><img src='data:image/png;base64,{}' class='img-fluid' width=50%/></a>".format(img_to_bytes('./icons/colab.png')), unsafe_allow_html=True)