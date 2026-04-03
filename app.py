import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import tempfile

st.title("🎵 Musical Instrument Classifier")

# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_model():
    from tensorflow.keras.models import load_model
    return load_model("instrument_classifier.h5")

model = load_model()

# ---------------- LABELS ----------------
classes = ['Piano', 'Guitar', 'Drums', 'Violin', 'Flute']

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(y, sr):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Resize to model input shape (adjust if needed)
    mel_db = np.resize(mel_db, (128, 128))
    mel_db = mel_db.reshape(1, 128, 128, 1)

    return mel_db

# ---------------- PLOTTING ----------------
def plot_waveform(y, sr):
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    return fig

def plot_spectrogram(y, sr):
    fig, ax = plt.subplots()
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)

    img = librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set_title("Mel Spectrogram")

    return fig

# ---------------- SEGMENT PREDICTION ----------------
def segment_prediction(y, sr):
    segment_duration = 2  # seconds
    step = sr * segment_duration

    results = []

    for i in range(0, len(y), step):
        segment = y[i:i+step]

        if len(segment) < step:
            continue

        features = extract_features(segment, sr)
        pred = model.predict(features, verbose=0)

        label = classes[np.argmax(pred)]
        confidence = np.max(pred)

        results.append((i/sr, label, confidence))

    return results

# ---------------- UI ----------------
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    y, sr = librosa.load(temp_path, duration=10)

    # AUDIO PLAYER
    st.audio(uploaded_file)

    # WAVEFORM
    st.subheader("📊 Waveform")
    fig1 = plot_waveform(y, sr)
    st.pyplot(fig1)

    # SPECTROGRAM
    st.subheader("🔥 Spectrogram")
    fig2 = plot_spectrogram(y, sr)
    st.pyplot(fig2)

    # OVERALL PREDICTION
    features = extract_features(y, sr)
    pred = model.predict(features, verbose=0)

    label = classes[np.argmax(pred)]
    confidence = np.max(pred)

    st.subheader("🎯 Overall Prediction")
    st.write(f"**Instrument:** {label}")
    st.write(f"**Confidence:** {confidence:.2f}")

    # SEGMENT TIMELINE
    st.subheader("⏱ Segment-wise Prediction")

    results = segment_prediction(y, sr)

    for time, label, conf in results:
        st.write(f"{time:.1f}s → {label} ({conf:.2f})")
