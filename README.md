# MOODIFY: Emotion-Based Music Recommendation System

Moodify is an AI-driven emotion recognition and music recommendation platform that uses a convolutional neural network (CNN) to classify user emotions from facial expressions and generate personalized playlists. The system enables users to upload images or capture real-time selfies, predicts the dominant emotion, and recommends music genres that align with the detected mood.

This project was developed as part of the Infosys Springboard Virtual Internship 6.0, where the objective was to build a personalized emotional music recommender using computer vision, machine learning, and modern application frameworks. The project journey, development stages, and outcomes are documented in the internship completion report submitted under the project title EmoTune.

---

## Dataset

The model was trained using the Balanced RAF-DB dataset sourced from Kaggle:

https://www.kaggle.com/datasets/dollyprajapati182/balanced-raf-db-dataset-7575-grayscale

Dataset characteristics:
- Real-world grayscale facial expression dataset
- Images resized to **75×75**
- Uniform distribution across 7 emotion labels: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Balanced and preprocessed for robust CNN training

The dataset exploration, preprocessing decisions, and resizing steps were completed during Week 1 and Week 2 of the internship, forming the foundation for the downstream model development.

---

## Features

- Facial emotion detection using a custom CNN model
- Photo upload and webcam capture for real-time inference
- Streamlit-based interactive UI and bottom music player
- Personalized music recommendations fetched using Jamendo and Audius APIs
- Firebase Authentication for secure login
- Firestore database integration for storing user profiles and mood history
- Genre mapping aligned to emotional states

---

## Tech Stack

**Frontend & UI**
- Streamlit
- HTML/CSS with cinematic theme customization

**Machine Learning & Vision**
- TensorFlow, Keras
- OpenCV for face detection, preprocessing, and grayscale enhancement

**Backend & Services**
- Firebase Firestore and Authentication
- Python-based server logic and API handling

---

## Directory Structure


```
├── app.py                      # Main application
├── Moodify.py                  # Additional module(s)
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── .gitignore                  # Git ignore rules
├── firebase.json               # Firebase hosting config
├── firestore.rules             # Firestore rules
├── firestore.indexes.json      # Firestore database indexes
├── public/                     # Firebase-auth hosted HTML pages
├── notebooks/                  # Model training and preprocessing notebooks
├── models/                     # Trained model (not pushed to GitHub)
├── Balanced RAF 75x75/         # Kaggle dataset (ignored)
├── preprocessed_data/          # Extracted numpy arrays and features
├── dataconnect/                # Data connector configs
├── img/                        # Static images
```

---

## Internship Context

This project originated from the Infosys Springboard Virtual Internship 6.0, Batch-2, starting on **13 October 2025**. The internship involved structured weekly milestones with guided development, culminating in a full-stack working prototype that demonstrates emotion-driven playlist generation.

The internship report outlines the weekly progression:
- Dataset analysis and image resizing (Week 1–2)
- CNN model development and accuracy tuning (Week 3–4)
- Music API integration and playlist logic (Week 5)
- UI and profile system implementation (Week 6)
- Testing, debugging, deployment, and documentation (Week 7–8)

All activities, including deliverables and screenshots of working modules, are recorded in the completion report.

---

## Installation


```bash
git clone https://github.com/Akashj-aj/Moodify.git
cd Moodify
pip install -r requirements.txt
cp .env.example .env  # then fill in your secrets
streamlit run app.py
```

---

## How It Works

1. User captures a photo or uploads an image.
2. Face is detected and preprocessed using OpenCV.
3. Input is passed to the CNN to classify emotion.
4. Emotion is mapped to suitable genres (e.g., Happy → Pop/Electronic).
5. Music recommendations are fetched dynamically from supported APIs.
6. The user plays music through the integrated audio interface.
7. Mood statistics and preferences are stored in Firebase Firestore.

---

## Future Enhancements

* Voice and text-based emotion understanding
* Improved mobile interface and responsive UI design
* Integration with commercial streaming services
* Adaptive learning based on user history

---

## Acknowledgements

The project was supported by Infosys Springboard's virtual internship program, which provided structured learning, weekly reviews, and domain guidance. The Balanced RAF dataset authors and contributors on Kaggle also enabled high-quality model training.

---

## License

This project is intended for academic, educational, and research use. Dataset and API usage is subject to third-party licensing. For code, see the LICENSE file (MIT).
