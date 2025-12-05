# Moodify

Moodify is an emotion-powered music discovery app built with Streamlit, Firebase, and deep learning. It matches your mood to music using facial emotion detection.

## Features
- 🎵 Discover music based on your emotions
- 📸 Upload a photo or use your webcam for emotion analysis
- 🔒 Secure authentication with Firebase
- 🟢 Cinematic dark UI inspired by Waveform Framer
- 🧠 Deep learning model for emotion recognition
- 📊 Mood history tracking

## Setup

1. **Clone the repository**
   ```sh
   git clone https://github.com/yourusername/moodify.git
   cd moodify
   ```
2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
3. **Create your `.env` file**
   - Copy `.env.example` to `.env`
   - Fill in your Firebase and Jamendo credentials

4. **Add your Firebase Admin SDK JSON**
   - Place your `firebase-adminsdk.json` in the project root (not tracked by git)

5. **Run the app**
   ```sh
   streamlit run app.py
   ```

## Deployment
- All secrets are loaded from `.env` and are not tracked by git.
- Do not commit your `.env` or `firebase-adminsdk.json`.

## Folder Structure
```
app.py
requirements.txt
.env.example
.gitignore
models/
preprocessed_data/
public/
notebooks/
Balanced RAF 75x75/
dataconnect/
img/
```

## License
MIT
