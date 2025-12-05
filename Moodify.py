import streamlit as st
import numpy as np
import cv2
import requests
import random
from tensorflow import keras

# CONFIG
st.set_page_config(page_title="Moodify - Emotion-Based Music Player", layout="centered")

# API CONFIG
JAMENDO_CLIENT_ID = "f4e409a8"
JAMENDO_BASE_URL = "https://api.jamendo.com/v3.0/tracks"
AUDIUS_API = "https://discoveryprovider.audius.co/v1/tracks/search"

# EMOTION LABELS
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# EMOTION → GENRES
emotion_to_genres = {
    "Happy": ["pop", "dance", "acoustic", "party", "electronic"],
    "Sad": ["melancholic", "soft", "piano", "romantic", "acoustic"],
    "Angry": ["rock", "metal", "energetic", "hard"],
    "Surprise": ["electronic", "upbeat", "funk", "pop"],
    "Fear": ["ambient", "dark", "cinematic", "dramatic"],
    "Disgust": ["grunge", "raw", "alternative", "industrial"],
    "Neutral": ["chill", "lofi", "calm", "indie"]
}

# LOAD MODEL
@st.cache_resource
def load_emotion_model():
    return keras.models.load_model("models/emotion_v2.keras")

model = load_emotion_model()

# FACE DETECTION
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#  Improved preprocessing 
def preprocess_image(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 75, 75)
    gray = cv2.medianBlur(gray, 3)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face = gray[y:y + h, x:x + w]
    else:
        face = gray
    face = cv2.resize(face, (75, 75))
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    face = clahe.apply(face)
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=(0, -1))
    return face


#  Stable multi-crop emotion prediction
def stable_predict_emotion(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    preds_list = []
    for shift in [-5, 0, 5]:
        M = np.float32([[1, 0, shift], [0, 1, 0]])
        shifted = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]))
        processed = preprocess_image(cv2.cvtColor(shifted, cv2.COLOR_GRAY2BGR))
        preds = model.predict(processed, verbose=0)[0]
        preds_list.append(preds)
    avg_preds = np.mean(preds_list, axis=0)
    return emotion_labels[np.argmax(avg_preds)]


# MUSIC FETCHERS
def get_tracks_jamendo(tag, limit=5, order="relevance"):
    """Fetch songs from Jamendo"""
    offset = random.randint(0, 200)
    url = (
        f"{JAMENDO_BASE_URL}?client_id={JAMENDO_CLIENT_ID}"
        f"&limit={limit}&fuzzytags={tag}"
        f"&audioformat=mp31&include=musicinfo"
        f"&order={order}&offset={offset}&format=json"
    )
    try:
        res = requests.get(url, timeout=10).json()
        if "results" in res and len(res["results"]) > 0:
            return [
                {
                    "id": t["id"],
                    "title": t["name"],
                    "artist": t["artist_name"],
                    "audio": t["audio"],
                    "image": t.get("image"),
                }
                for t in res["results"]
                if t.get("audio")
            ]
    except Exception:
        pass
    return []


def get_tracks_audius(keywords, limit=5):
    """Backup: fetch songs from Audius (multi-genre capable)"""
    query = " ".join(keywords) if isinstance(keywords, list) else keywords
    try:
        params = {"query": query, "app_name": "moodify_app", "limit": limit}
        res = requests.get(AUDIUS_API, params=params, timeout=10).json()
        if "data" in res and len(res["data"]) > 0:
            return [
                {
                    "id": t["id"],
                    "title": t["title"],
                    "artist": t["user"]["name"],
                    "audio": t.get("stream_url"),
                    "image": None,
                    "source": "Audius"
                }
                for t in res["data"]
                if t.get("stream_url")
            ]
    except Exception:
        pass
    return []


#  Smart genre merging 
def combine_genres(genres):
    if not genres:
        return "pop"
    gset = set(genres)
    mapping_rules = {
        frozenset(["pop", "dance"]): "electronic",
        frozenset(["rock", "metal"]): "hard",
        frozenset(["acoustic", "piano", "romantic"]): "melancholic",
        frozenset(["ambient", "chill", "calm"]): "lofi",
        frozenset(["grunge", "alternative", "industrial"]): "raw",
        frozenset(["electronic", "funk", "upbeat"]): "party",
    }
    for key, value in mapping_rules.items():
        if key.issubset(gset):
            return value
    priority = ["electronic", "pop", "rock", "metal", "piano", "acoustic", "lofi", "romantic"]
    for p in priority:
        if p in gset:
            return p
    return genres[0]


# SESSION STATE
if "previous_songs" not in st.session_state:
    st.session_state.previous_songs = set()
if "songs_displayed" not in st.session_state:
    st.session_state.songs_displayed = False


def display_songs(songs):
    """Display songs while avoiding repeats"""
    st.session_state.songs_displayed = True
    for s in songs:
        if s["id"] not in st.session_state.previous_songs:
            st.session_state.previous_songs.add(s["id"])
            if s.get("image"):
                st.image(s["image"], width=150)
            st.markdown(f"### 🎶 {s['title']}")
            st.write(f"**Artist:** {s['artist']}")
            st.audio(s["audio"], format="audio/mp3")
            st.write("---")


# FETCH + DISPLAY 
def fetch_and_display(selected_genres, emotion, song_mode, shuffle=False):
    tag = combine_genres(selected_genres)
    order = "relevance" if song_mode == "Random Songs" else "popularity_total"

    songs = get_tracks_jamendo(tag, order=order)
    if not songs:
        st.info("No Jamendo results — fetching from Audius instead ")
        songs = get_tracks_audius(selected_genres)

    if shuffle:
        random.shuffle(songs)

    if songs:
        st.info(f"Fetching {emotion}-mood {song_mode.lower()}...")
        display_songs(songs)
    else:
        st.warning("No songs found ")


# UI
st.title("Moodify")
st.markdown("### _Let your emotion choose your music!_")

mode = st.radio("Choose input mode:", ["Upload Image", "Webcam (Take Photo)"], index=0)

# UPLOAD MODE
if mode == "Upload Image":
    uploaded = st.file_uploader("Upload your face image:", type=["jpg", "jpeg", "png"])
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", width=300)

        emotion = stable_predict_emotion(img)
        st.success(f"**Detected Emotion:** {emotion}")

        genres = emotion_to_genres[emotion]
        selected_genres = st.multiselect("Select preferred genres:", genres, default=[genres[0]])

        song_mode = st.selectbox("Choose Song Mode:", ["Random Songs", "Most Liked Songs"])

        if st.button("Fetch Songs"):
            fetch_and_display(selected_genres, emotion, song_mode, shuffle=False)

        if st.session_state.songs_displayed:
            if st.button("Shuffle Songs"):
                fetch_and_display(selected_genres, emotion, song_mode, shuffle=True)


# WEBCAM MODE
elif mode == "Webcam (Take Photo)":
    camera_file = st.camera_input("Take a photo")
    if camera_file is not None:
        file_bytes = np.asarray(bytearray(camera_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        display_img = cv2.flip(img, 1)
        st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), caption="Captured", width=600)

        emotion = stable_predict_emotion(img)
        st.success(f"**Detected Emotion:** {emotion}")

        genres = emotion_to_genres[emotion]
        selected_genres = st.multiselect("Select preferred genres:", genres, default=[genres[0]])

        song_mode = st.selectbox("Choose Song Mode:", ["Random Songs", "Most Liked Songs"])

        if st.button("Fetch Songs"):
            fetch_and_display(selected_genres, emotion, song_mode, shuffle=False)

        if st.session_state.songs_displayed:
            if st.button("Shuffle Songs"):
                fetch_and_display(selected_genres, emotion, song_mode, shuffle=True)


st.markdown("---")
st.caption(" Moodify — Emotion-based Music Player ")
