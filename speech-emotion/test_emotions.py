import sys
import numpy as np
import tensorflow as tf
import soundfile as sf
import pickle
from scipy.fftpack import dct
from tensorflow.keras.models import load_model
import os

# Load trained model
model = load_model("speech_emotion_model.h5")

# Load LabelEncoder
try:
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    print("Label encoder loaded successfully.")
except FileNotFoundError:
    print("Error: label_encoder.pkl not found. Train the model first.")
    sys.exit(1)

# Define function to extract MFCC features
def extract_features(file_path, max_pad_len=100, num_mfcc=40):
    try:
        audio, sr = sf.read(file_path)
        if len(audio) == 0:
            print(f"Warning: {file_path} is empty.")
            return np.zeros((num_mfcc, max_pad_len))

        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)  

        frame_length = int(0.025 * sr)
        frame_step = int(0.01 * sr)
        num_frames = max(1, (len(audio) - frame_length) // frame_step + 1)

        frames = np.array([audio[i * frame_step: i * frame_step + frame_length] for i in range(num_frames)])
        power_spectrum = np.abs(np.fft.rfft(frames, axis=-1)) ** 2
        log_spectrum = np.log(np.maximum(power_spectrum, 1e-10))
        mfcc = dct(log_spectrum, type=2, axis=-1, norm='ortho')[:, :num_mfcc]

        pad_width = max_pad_len - mfcc.shape[0]
        if pad_width > 0:
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:max_pad_len, :]

        return mfcc.T
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros((num_mfcc, max_pad_len))

# Function to predict emotion
def predict_emotion(audio_file):
    features = extract_features(audio_file)
    features = np.expand_dims(features, axis=0)

    prediction = model.predict(features)
    predicted_label = np.argmax(prediction)

    # Debugging: print predicted index and probabilities
    print("Raw Prediction Probabilities:", prediction)
    print("Predicted Label Index:", predicted_label)

    # Convert label index to emotion
    if predicted_label >= len(label_encoder.classes_):
        print(f"Error: Predicted label {predicted_label} is out of range.")
    else:
        print(f"Predicted Emotion: {label_encoder.inverse_transform([predicted_label])[0]}")

# Run script
if len(sys.argv) < 2:
    print("Usage: python test_emotion.py <path_to_audio.wav>")
    sys.exit(1)

audio_file = sys.argv[1]

if not os.path.exists(audio_file):
    print(f"Error: File '{audio_file}' not found.")
    sys.exit(1)

predict_emotion(audio_file)
