# ------------------- SILENCE WARNINGS -------------------
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import streamlit as st
import numpy as np
import cv2
import requests
import random
import base64
import json
from streamlit import query_params
import firebase_admin
from firebase_admin import credentials, firestore
from tensorflow import keras
from dotenv import load_dotenv
load_dotenv()

import time

# ------------------- CINEMATIC DARK THEME + PERFECT STREAMLIT NAVBAR -------------------
st.set_page_config(page_title="Moodify", layout="wide")
st.markdown("""
<style>
    /* ───── CINEMATIC BACKGROUND (unchanged) ───── */
    .cinema-bg::before {
        content: ""; position: fixed; top: -300px; left: 50%; transform: translateX(-50%);
        width: 2800px; height: 1600px;
        background: radial-gradient(ellipse at center, rgba(255,255,255,0.22) 0%, rgba(255,255,255,0.12) 25%, transparent 70%);
        filter: blur(80px); opacity: 0.9; pointer-events: none; z-index: 1;
    }
    .cinema-bg::after {
        content: ""; position: fixed; top: -2500px; left: 50%; transform: translateX(-50%);
        width: 4000px; height: 3500px;
        background:
            repeating-conic-gradient(from 0deg at 50% 50%, rgba(255,255,255,0.07) 0deg 0.9deg, transparent 0.9deg 18deg),
            repeating-conic-gradient(from 15deg at 50% 50%, rgba(255,255,255,0.04) 0deg 0.7deg, transparent 0.7deg 20deg);
        opacity: 0.7; filter: blur(2px);
        animation: spin 380s linear infinite; pointer-events: none; z-index: 0;
    }
    @keyframes spin {from {transform: translateX(-50%) rotate(0deg)} to {transform: translateX(-50%) rotate(360deg)}}
    /* ───── REMOVE ALL DEFAULT STREAMLIT SPACING ───── */
    #MainMenu, header, footer, [data-testid="stSidebar"], [data-testid="collapsedControl"], section[data-testid="stDecoration"] {display: none !important;}
    .block-container {padding-top: 0rem !important; margin-top: 0rem !important;}
    .stApp {background: #000 !important; margin: 0; padding: 0;}
            
    /* FULLY REMOVE ALL SIDE PADDING — TRUE FULLSCREEN CINEMATIC EXPERIENCE */
    [data-testid="stAppViewContainer"] > .st-emotion-cache-zy6yx3,
    [data-testid="stAppViewContainer"] .st-emotion-cache-zy6yx3 {
        padding-left: 8px !important;
        padding-right: 8px !important;
        margin-left: 0px !important;
        margin-right: 0px !important;
        max-width: none !important;
        padding-bottom: 0px !important;
        margin-bottom: 0px !important;
    }
    /* Also kill any potential inner containers trying to add padding */
    section.main > div:first-child,
    section.main > .block-container {
        padding-left: 0px !important;
        padding-right: 0px !important;
        max-width: 100vw !important;
    }
    /* Ensure the root app takes full width */
    .stApp {
        margin: 0 !important;
        padding: 0 !important;
        max-width: 100vw !important;
        overflow-x: hidden;
    }
    /* ───── PERFECT GLASSMORPHIC NAVBAR BUTTONS (2025-proof) ───── */
    /* Target ALL buttons in the navbar — works reliably in Streamlit 1.38+ */
    .stButton button[kind="secondary"] {
        all: unset !important;
        font-family: "SF Pro Display", -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        color: rgba(255, 255, 255, 0.85) !important;
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1.4px solid rgba(255, 255, 255, 0.18) !important;
        border-radius: 50px !important;
        padding: 10px 18px !important;
        min-height: 30px !important;
        min-width: 100px !important;
        text-align: center !important;
        cursor: pointer !important;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow:
            0 8px 25px rgba(0, 0, 0, 0.4),
            0 0 20px rgba(29, 185, 84, 0.12) !important;
        margin: 0 10px !important;
        position: relative !important;
        z-index: 2;
    }
    /* Hover — lift + glowing green border */
    .stButton button[kind="secondary"]:hover {
        background: rgba(255, 255, 255, 0.16) !important;
        border-color: #1db954 !important;
        color: white !important;
        transform: translateY(-4px) scale(1.03) !important;
        box-shadow:
            0 16px 40px rgba(0, 0, 0, 0.5),
            0 0 30px rgba(29, 185, 84, 0.45) !important;
    }
    /* Active / Currently Selected Page — STRONG GREEN GLOW (like Spotify) */
    div[data-testid="stVerticalBlock"] div.stButton > button[kind="secondary"]:focus,
    div[data-testid="stVerticalBlock"] div.stButton > button[kind="secondary"]:active,
    div[data-testid="stVerticalBlock"] div.stButton > button[kind="secondary"][aria-pressed="true"] {
        background: linear-gradient(135deg, rgba(29, 185, 84, 0.55), rgba(29, 185, 84, 0.35)) !important;
        border: 2px solid #1db954 !important;
        color: white !important;
        font-weight: 700 !important;
        transform: translateY(-4px) !important;
        box-shadow: 
            0 0 40px rgba(29, 185, 84, 0.9),
            0 0 80px rgba(29, 185, 84, 0.6),
            inset 0 0 20px rgba(29, 185, 84, 0.4) !important;
        animation: glow-pulse 4s ease-in-out infinite !important;
    }
    /* Optional: Remove default focus outline */
    @keyframes glow-pulse {
        0%, 100% { box-shadow: 0 0 40px rgba(29,185,84,0.9), 0 0 80px rgba(29,185,84,0.6); }
        50%      { box-shadow: 0 0 50px rgba(29,185,84,1),   0 0 100px rgba(29,185,84,0.8); }
    }
    /* ───── LOGO & PROFILE (enhanced) ───── */
    .nav-logo h1 {
        font-size: 52px !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #ffffff 0%, #a8ffc9 50%, #1db954 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        letter-spacing: 8px !important;
        text-shadow: 0 0 60px rgba(29, 185, 84, 0.7) !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .nav-profile {
        display: flex;
        align-items: center;
        gap: 12px;
        color: rgba(255,255,255,0.9);
        font-weight: 500;
    }
    .nav-profile img {
        width: 56px !important;
        height: 56px !important;
        border-radius: 50% !important;
        border: 3.5px solid #1db954 !important;
        box-shadow: 0 0 50px rgba(29, 185, 84, 0.9) !important;
        object-fit: cover;
    }
    /* Navbar layout breathing */
    .moodify-navbar {
        padding: 2rem 1rem 1rem !important;
        display: flex;
        align-items: center;
        justify-content: space-between;
        position: relative;
        z-index: 10;
    }
    div[data-testid="stVerticalBlock"] > div.st-emotion-cache-tn0cau,
    div.st-emotion-cache-tn0cau {
        gap: 0rem !important;
    }
    /* Remove Streamlit's hidden side padding once and for all */
section.main > .block-container {
    padding-left: 0 !important;
    padding-right: 0 !important;
    max-width: none !important;
}
/* ───── STICKY BOTTOM PLAYER ───── */
.main-content {
    padding-bottom: 0px;
}
.player-bar {
    position: fixed;
    bottom: 0px;
    left: 0px;
    width: 100%;
    height: 73px;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    padding: 8px 25px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    z-index: 1000;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.5);
}
.wave-container {
    display: flex;
    align-items: end;
    height: 28px;
    gap: 2px;
    flex: 1;
    justify-content: center;
    max-width: 300px;
}
.wave-bar {
    width: 4px;
    background: linear-gradient(to top, #1db954, #4fc3a1);
    border-radius: 2px 2px 0 0;
    animation: wave 3s cubic-bezier(0.25, 0.46, 0.45, 0.94) infinite;
    flex: none;
}
@keyframes wave {
    0% {
        height: 3px;
        transform: scaleY(1);
    }
    25% {
        height: 10px;
        transform: scaleY(1.3);
    }
    50% {
        height: 25px;
        transform: scaleY(2.5);
    }
    75% {
        height: 14px;
        transform: scaleY(1.8);
    }
    100% {
        height: 3px;
        transform: scaleY(1);
    }
}
    /* Hover effect for discover cards */
    div[data-testid="column"] > div > div > div > div:hover {
        transform: translateY(-10px) scale(1.02) !important;
        box-shadow: 0 20px 50px rgba(29, 185, 84, 0.4), 0 0 40px rgba(29, 185, 84, 0.25) !important;
        border-color: #1db954 !important;
    }
    /* Spotify-like playlist row styling */
    .playlist-row {
        display: flex;
        align-items: center;
        padding: 12px 16px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        margin-bottom: 4px;
        transition: background 0.2s;
    }
    .playlist-row:hover {
        background: rgba(255, 255, 255, 0.1);
    }
    .playlist-row.playing {
        background: linear-gradient(90deg, rgba(29, 185, 84, 0.2), rgba(255, 255, 255, 0.05));
        border-left: 4px solid #1db954;
    }
    /* REDUCE WEBCAM SIZE  */
    [data-testid="stCameraInput"] {
        width: 1100px !important;      
        height: 750px !important;     
        max-width: 100vw !important;
        margin: 2rem auto !important; 
        border-radius: 16px !important;
        overflow: hidden;
        border: 3px solid rgba(29, 185, 84, 0.4) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6) !important;
    }

    [data-testid="stCameraInput"] video {
        width: 100% !important;
        height: 100% !important;
        object-fit: cover !important;
    }
    </style>
<div class="cinema-bg"></div>
""", unsafe_allow_html=True)
# ------------------- INIT SESSION STATE -------------------
if "initialized" not in st.session_state:
    st.session_state.update({
        "user": None, "username": "", "email": "", "profile_pic": "",
        "current_playlist": [], "current_song": None,
        "page": "Home", "last_emotion": None, "selected_genres": [], "initialized": True
    })
# ------------------- FIREBASE & MODEL -------------------
db = None
try:
    if not firebase_admin._apps:
        cred_path = os.getenv("FIREBASE_ADMINSDK_PATH", "firebase-adminsdk.json")
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
    db = firestore.client()
except: db = None
model = None
try:
    @st.cache_resource
    def load_emotion_model():
        return keras.models.load_model("models/emotion_v2.keras")
    model = load_emotion_model()
except: model = None
emotion_labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
emotion_to_genres = {
    "Happy": ["pop", "dance", "acoustic", "party", "electronic"],
    "Sad": ["melancholic", "soft", "piano", "romantic", "acoustic"],
    "Angry": ["rock", "metal", "energetic", "hard"],
    "Surprise": ["electronic", "upbeat", "funk", "pop"],
    "Fear": ["ambient", "dark", "cinematic", "dramatic"],
    "Disgust": ["grunge", "raw", "alternative", "industrial"],
    "Neutral": ["chill", "lofi", "calm", "indie"]
}
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
def preprocess_image(img_bgr):
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 5, 75, 75)
        gray = cv2.medianBlur(gray, 3)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x,y,w,h = max(faces, key=lambda r: r[2]*r[3])
            face = gray[y:y+h, x:x+w]
        else:
            face = gray
        face = cv2.resize(face, (75,75))
        face = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8)).apply(face)
        face = face.astype("float32")/255.0
        face = np.expand_dims(face, axis=(0,-1))
        return face
    except: return np.zeros((1,75,75,1))

def stable_predict_emotion(img_bgr):
    if model is None: return "Neutral"
    try:
        preds = []
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        for shift in [-5,0,5]:
            M = np.float32([[1,0,shift],[0,1,0]])
            shifted = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]))
            processed = preprocess_image(cv2.cvtColor(shifted, cv2.COLOR_GRAY2BGR))
            preds.append(model.predict(processed, verbose=0)[0])
        return emotion_labels[np.argmax(np.mean(preds, axis=0))]
    except: return "Neutral"

# ------------------- MUSIC FETCHING -------------------

JAMENDO_CLIENT_ID = os.getenv("JAMENDO_CLIENT_ID", "")
def get_tracks_jamendo(tag, limit=15):
    offset = random.randint(0, 300)
    url = f"https://api.jamendo.com/v3.0/tracks?client_id={JAMENDO_CLIENT_ID}&format=json&limit={limit}&fuzzytags={tag}&audioformat=mp31&include=musicinfo&offset={offset}"
    try:
        data = requests.get(url, timeout=10).json()
        if data.get("results"):
            return [{"id":t["id"],"title":t["name"],"artist":t["artist_name"],"audio":t["audio"],
                     "image":t.get("image") or "https://via.placeholder.com/150"} 
                    for t in data["results"] if t.get("audio")]
    except: pass
    return []
def get_tracks_audius(genres, limit=15):
    query = " ".join(genres)
    try:
        res = requests.get(f"https://discoveryprovider.audius.co/v1/tracks/search?query={query}&app_name=moodify", timeout=10).json()
        if res.get("data"):
            return [{"id":t["id"],"title":t["title"],"artist":t["user"]["name"],"audio":t["stream_url"],
                     "image":None} for t in res["data"][:limit] if t.get("stream_url")]
    except: pass
    return []
def combine_genres(genres):
    if not genres: return "pop"
    rules = {frozenset(["pop","dance"]):"electronic", frozenset(["rock","metal"]):"hard rock",
             frozenset(["acoustic","piano","romantic"]):"sad", frozenset(["chill","calm","ambient"]):"lofi"}
    s = set(genres)
    for k,v in rules.items():
        if k.issubset(s): return v
    return genres[0]
# ------------------- AUTH -------------------
params = query_params.to_dict()
def get_param(key):
    val = params.get(key)
    return val[0] if isinstance(val, list) else val
if st.session_state.user is None and "auth" in params:
    try:
        payload = json.loads(base64.b64decode(get_param("auth")).decode())
        uid = payload["uid"]
        email = payload["email"]
        name = payload.get("displayName") or email.split("@")[0]
        photo = payload.get("photoURL", "https://via.placeholder.com/48/333/fff?text=U")
        if db:
            ref = db.collection("users").document(uid)
            if not ref.get().exists:
                ref.set({"uid": uid, "email": email, "username": name, "profile_pic": photo,
                         "created_at": firestore.SERVER_TIMESTAMP,
                         "mood_history": {e.lower():0 for e in emotion_labels}})
            data = ref.get().to_dict()
            st.session_state.update({"user": uid, "username": data.get("username", name), "email": email, "profile_pic": data.get("profile_pic", photo)})
        else:
            st.session_state.update({"user": uid, "username": name, "email": email, "profile_pic": photo})
        query_params.clear()
        st.success(f"Welcome, {name.split()[0]}!")
        st.rerun()
    except Exception as e:
        st.error("Login failed")
        st.stop()
if st.session_state.user is None:
    st.markdown("""
    <meta http-equiv="refresh" content="0; url=https://studio-5934774658-e4401.web.app/index.html">
    <h1 style="text-align:center;color:#1db954;font-size:80px;margin-top:40vh;">MOODIFY</h1>
    """, unsafe_allow_html=True)
    st.stop()
# ------------------- BEAUTIFUL STREAMLIT NAVBAR -------------------
st.markdown('<div class="moodify-navbar">', unsafe_allow_html=True)
left, center, right = st.columns([4, 12, 2], gap="large")
with left:
    st.markdown('<div class="nav-logo"><h1>MOODIFY</h1></div>', unsafe_allow_html=True)
    
with center:
    st.markdown('<div style="display:flex; justify-content:center; gap:16px; flex-wrap:wrap;">', unsafe_allow_html=True)
    b1, b2, b3, b4, b5 = st.columns([1,1,1,1,1])
    with b1:
        if st.button("Home", key="nav_home"):
            st.session_state.page = "Home"
            st.rerun()
    with b2:
        if st.button("Upload Photo", key="nav_upload_photo"):
            st.session_state.page = "Upload Photo"
            st.rerun()
    with b3:
        if st.button("Webcam Capture", key="nav_webcam_capture"):
            st.session_state.page = "Webcam Capture"
            st.rerun()
    with b4:
        if st.button("Mood History", key="nav_mood_history"):
            st.session_state.page = "Mood History"
            st.rerun()
    with b5:
        if st.button("Profile", key="nav_profile"):
            st.session_state.page = "Profile"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
with right:
    st.markdown(f'''
    <div class="nav-profile">
        <span>{st.session_state.username}</span>
        <img src="{st.session_state.profile_pic or 'https://via.placeholder.com/48/333/fff?text=U'}">
    </div>
    ''', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ------------------- MAIN CONTENT -------------------
st.markdown('<div class="main-content">', unsafe_allow_html=True)
# Layout: Main Content + Right Player Panel
col_main, col_player = st.columns([5, 1], gap="large")  

with col_main:
    main_container = st.container(height=550)

    with main_container:
        if st.session_state.page == "Home":
            st.markdown("""
            <style>
            /* Force this exact container to be the only scrollable part */
            [data-testid="stVerticalBlock"] > div:first-child {
                padding: 0 !important;
            }
            </style>
            """, unsafe_allow_html=True)

            @st.cache_data(ttl=3600, show_spinner=False)
            def get_discover_songs():
                tags = ["pop", "electronic", "chill", "indie", "rock", "acoustic", "dance", "lofi"]
                all_songs = []
                for tag in random.sample(tags, k=4):
                    songs = get_tracks_jamendo(tag, limit=8) or get_tracks_audius([tag], limit=8)
                    all_songs.extend(songs)
                random.shuffle(all_songs)
                return all_songs[:12]  # Only 12 songs

            discover_songs = get_discover_songs()

            st.markdown("<h2 style='text-align:center; color:#1db954; margin:2rem 0 1.5rem 0;'>Discover Today</h2>", 
                        unsafe_allow_html=True)

            if not discover_songs:
                st.info("Loading fresh tracks for you...")
            else:
                cols = st.columns(4)
                for idx, song in enumerate(discover_songs):
                    with cols[idx % 4]:
                        st.markdown(f"""
                        <div style="background:rgba(255,255,255,0.07); border-radius:16px; overflow:hidden;
                                    border:1px solid rgba(255,255,255,0.12); cursor:pointer;
                                    transition:all 0.3s; margin-bottom:1.5rem; width: 250px; margin-left: auto; margin-right: auto;
                                    box-shadow:0 8px 32px rgba(0,0,0,0.4);">
                            <img src="{song.get('image') or 'https://via.placeholder.com/250/1db954/000?text=♪'}"
                                 style="width:100%; aspect-ratio:1/1; object-fit:cover;">
                            <div style="padding:12px;">
                                <div style="font-weight:700; font-size:14px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                                    {song['title']}
                                </div>
                                <div style="color:rgba(255,255,255,0.7); font-size:12px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                                    {song['artist']}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        _, center_col, _ = st.columns([0.5, 1.2, 0.7])
                        with center_col:
                            if st.button("Play Now", key=f"home_play_{song['id']}_{idx}"):
                                st.session_state.current_song = song
                                playlist = st.session_state.current_playlist or []
                                # Put current song first
                                new_playlist = [song] + [s for s in playlist if s["id"] != song["id"]]
                                st.session_state.current_playlist = new_playlist[:50]
                                st.rerun()

        elif st.session_state.page in ["Upload Photo", "Webcam Capture"]:
            st.markdown(f"### Feeling **{st.session_state.get('last_emotion', '?')}** Today?")
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            img = None
            if st.session_state.page == "Upload Photo":
                uploaded = st.file_uploader("Upload a clear selfie", type=["jpg","jpeg","png"])
                if uploaded:
                    img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), 1)
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=400)
            else:
                pic = st.camera_input("Take a selfie")
                if pic:
                    img = cv2.imdecode(np.frombuffer(pic.read(), np.uint8), 1)
                    display_img = cv2.flip(img, 1)
                    st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), width=420)
            st.markdown("</div>", unsafe_allow_html=True)
            if img is not None:
                with st.spinner("Analyzing your mood..."):
                    emotion = stable_predict_emotion(img)
                    st.session_state.last_emotion = emotion
                    st.success(f"You're feeling **{emotion}**")
                    if db:
                        db.collection("users").document(st.session_state.user).update({f"mood_history.{emotion.lower()}": firestore.Increment(1)})
                    genres = emotion_to_genres[emotion]
                    selected = st.multiselect("Fine-tune genres", genres, default=genres[:3])
                    if st.button("Generate Playlist", type="primary"):
                        with st.spinner("Finding perfect tracks..."):
                            st.session_state.selected_genres = selected
                            tag = combine_genres(selected)
                            songs = get_tracks_jamendo(tag) or get_tracks_audius(selected)
                            if songs:
                                st.session_state.current_playlist = songs
                                st.session_state.current_song = songs[0]
                                st.success(f"Found {len(songs)} tracks for {emotion} mood!")
                                st.rerun()
            if st.session_state.current_playlist:
                st.markdown("### Your Mood Playlist")
                col_shuffle, _ = st.columns([1, 10])
                with col_shuffle:
                    if st.button("Shuffle", key="shuffle_playlist"):
                        if st.session_state.last_emotion:
                            with st.spinner("Fetching fresh tracks for your mood..."):
                                emotion = st.session_state.last_emotion
                                selected = st.session_state.get('selected_genres', emotion_to_genres[emotion][:3])
                                tag = combine_genres(selected)
                                new_songs = get_tracks_jamendo(tag) or get_tracks_audius(selected)
                                if new_songs:
                                    st.session_state.current_playlist = new_songs
                                    st.session_state.current_song = new_songs[0]
                                    st.success(f"Refreshed {len(new_songs)} new tracks for {emotion} mood!")
                                else:
                                    st.warning("Couldn't fetch new tracks, shuffling existing ones.")
                                    random.shuffle(st.session_state.current_playlist)
                                    st.session_state.current_song = st.session_state.current_playlist[0]
                        else:
                            random.shuffle(st.session_state.current_playlist)
                            st.session_state.current_song = st.session_state.current_playlist[0]
                        st.rerun()
                for idx, song in enumerate(st.session_state.current_playlist):
                    is_playing = st.session_state.current_song and st.session_state.current_song["id"] == song["id"]
                    # Spotify-like row layout
                    cols = st.columns([0.05, 0.08, 0.77, 0.1])
                    with cols[0]:
                        st.markdown(f'<div style="font-weight:600; color:rgba(255,255,255,0.6); text-align:center;">{idx+1 if not is_playing else "♪"}</div>', unsafe_allow_html=True)
                    with cols[1]:
                        st.image(song.get("image") or "https://via.placeholder.com/48/1db954/000000?text=Music", width=48)
                    with cols[2]:
                        col_title, col_artist = st.columns([1, 1])
                        with col_title:
                            st.markdown(f"**{song['title']}**")
                        with col_artist:
                            st.markdown(f"<small style='color:rgba(255,255,255,0.7);'>{song['artist']}</small>", unsafe_allow_html=True)
                    with cols[3]:
                        if st.button("▶", key=f"play_{song['id']}", help="Play"):
                            st.session_state.current_song = song
                            st.rerun()
        elif st.session_state.page == "Mood History":
            st.markdown("### Your Mood Journey")
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            if db:
                data = db.collection("users").document(st.session_state.user).get().to_dict()
                history = data.get("mood_history", {})
                total = sum(history.values())
                for mood, count in sorted(history.items(), key=lambda x: x[1], reverse=True):
                    if count > 0:
                        percent = (count / total * 100) if total > 0 else 0
                        st.markdown(f"**{mood.title()}** — {count} times ({percent:.1f}%)")
                        st.progress(percent / 100)
            else:
                st.info("Mood history disabled")
            st.markdown("</div>", unsafe_allow_html=True)
        elif st.session_state.page == "Profile":
            st.markdown("### Profile Settings")
            
            user_ref = db.collection("users").document(st.session_state.user) if db else None

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("#### Profile Picture")

                # CURRENT PICTURE
                current_pic = st.session_state.get("profile_pic") or "https://via.placeholder.com/200/333/fff?text=U"
                st.markdown(f"""
                <div style="width:180px;height:180px;border-radius:50%;overflow:hidden;
                            border:5px solid #1db954;box-shadow:0 0 60px rgba(29,185,84,0.6);
                            margin:20px auto;background:#111;">
                    <img src="{current_pic}" style="width:100%;height:100%;object-fit:cover;">
                </div>
                <p style="text-align:center;color:rgba(255,255,255,0.7);font-size:14px;">Current Picture</p>
                """, unsafe_allow_html=True)

                # FORCE CLEAR GHOST PREVIEW ON PAGE LOAD
                if st.session_state.get("clear_profile_preview", False):
                    for key in ["temp_profile_pic", "profile_pic_processed", "clear_profile_preview"]:
                        st.session_state.pop(key, None)
                    st.rerun()

                # UPLOAD METHOD
                pic_method = st.radio("Change picture via", ["Upload Image", "Image URL"], horizontal=True, key="pic_method_change")

                # === UPLOAD LOGIC (Upload Image) ===
                if pic_method == "Upload Image":
                    uploaded = st.file_uploader("Choose new picture", type=["png", "jpg", "jpeg"], key="profile_uploader", label_visibility="collapsed")
                    if uploaded:
                        bytes_data = uploaded.read()
                        b64 = base64.b64encode(bytes_data).decode()
                        st.session_state.temp_profile_pic = f"data:{uploaded.type};base64,{b64}"

                # === IMAGE URL (now accepts normal URLs + base64 data URLs) ===
                else:
                    url_input = st.text_input(
                        "Enter image URL or paste base64 data URL",
                        placeholder="https://example.com/pic.jpg  or  data:image/...",
                        key="url_input"
                    )
                    
                    if url_input:
                        url = url_input.strip()
                        if url.startswith(("http://", "https://", "data:image/")):
                            if url.startswith("http"):
                                try:
                                    r = requests.head(url, timeout=6, allow_redirects=True)
                                    content_type = r.headers.get("content-type", "")
                                    if r.status_code < 400 and ("image" in content_type or "octet-stream" in content_type):
                                        st.session_state.temp_profile_pic = url
                                    else:
                                        st.error("URL is not reachable or not an image")
                                except:
                                    st.error("Cannot reach this URL")
                            else:
                                # It's a base64 data URL → accept instantly
                                st.session_state.temp_profile_pic = url
                        else:
                            st.error("Invalid URL — must start with http://, https:// or data:image/")

                # === PREVIEW + SAVE/CANCEL BUTTONS ===
                if "temp_profile_pic" in st.session_state:
                    st.markdown(f"""
                    <div style="width:180px;height:180px;border-radius:50%;overflow:hidden;
                                border:6px solid #1db954;box-shadow:0 0 90px rgba(29,185,84,1);
                                margin:30px auto;background:#111;animation:glow-pulse 2s infinite;">
                        <img src="{st.session_state.temp_profile_pic}" style="width:100%;height:100%;object-fit:cover;">
                    </div>
                    <p style="text-align:center;color:#1db954;font-weight:700;font-size:18px;">Preview</p>
                    """, unsafe_allow_html=True)

                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("Save Picture", type="primary", use_container_width=True):
                            st.session_state.profile_pic = st.session_state.temp_profile_pic
                            if user_ref:
                                user_ref.update({"profile_pic": st.session_state.temp_profile_pic})
                            st.success("Profile picture updated!")
                            st.session_state.clear_profile_preview = True
                            st.rerun()

                    with c2:
                        if st.button("Cancel", use_container_width=True):
                            st.session_state.clear_profile_preview = True
                            st.rerun()

                else:
                    st.markdown("<p style='text-align:center;color:rgba(255,255,255,0.5);margin-top:20px;'>Choose or paste a new picture</p>", unsafe_allow_html=True)

            # ——————— RIGHT COLUMN: USERNAME & LOGOUT ———————
            with col2:
                st.markdown("#### Account Details")
                new_name = st.text_input("Username", value=st.session_state.username, key="username_edit")
                if new_name != st.session_state.username:
                    if st.button("Save Username", type="primary"):
                        st.session_state.username = new_name
                        if user_ref:
                            user_ref.update({"username": new_name})
                        st.success("Username updated!")
                        st.rerun()

                st.text_input("Email", value=st.session_state.email, disabled=True)
                st.markdown("---")
                if st.button("Logout", type="primary"):
                    st.session_state.clear()
                    st.rerun()
# Right Player Panel
with col_player:
    if st.session_state.current_song:
        song = st.session_state.current_song
        st.markdown('<h3 style="color:#1db954;font-size:22px;margin-bottom:16px;">Now Playing </h3>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.image(song.get("image") or "https://via.placeholder.com/300/1db954/000000?text=Music", width=300)
        st.markdown(f"**{song['title']}**")
        st.caption(f"by {song['artist']}")
        st.audio(song["audio"], format="audio/mp3")
st.markdown('</div>', unsafe_allow_html=True)
# ------------------- BOTTOM PLAYER -------------------
if st.session_state.current_song:
    song = st.session_state.current_song
    wave_bars = "".join([f'<div class="wave-bar" style="animation-delay: {i * 0.14}s;"></div>' for i in range(11)])

    st.markdown(f"""
    <div class="player-bar">
        <div style="display:flex;align-items:center;gap:18px;flex:0 0 auto;">
            <img src="{song.get('image') or 'https://via.placeholder.com/64'}" 
                 width="58" height="58" 
                 style="border-radius:10px;box-shadow:0 6px 20px rgba(0,0,0,0.6);flex-shrink:0;">
            <div style="min-width:0;">
                <div style="font-weight:700;font-size:16px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:280px;">
                    {song['title']}
                </div>
                <div style="color:rgba(255,255,255,0.8);font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:280px;">
                    {song['artist']}
                </div>
            </div>
        </div>
        <div class="wave-container">
            {wave_bars}
        </div>
    </div>
    """, unsafe_allow_html=True)
st.caption("")