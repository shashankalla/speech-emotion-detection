import os
import numpy as np
import tensorflow as tf
import soundfile as sf
from scipy.fftpack import dct
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pickle

# Define emotion mapping based on RAVDESS dataset filename format
emotion_map = {
    "01": "Neutral", "02": "Calm", "03": "Happy", "04": "Sad",
    "05": "Angry", "06": "Fearful", "07": "Disgust", "08": "Surprised"
}

# Define function to extract MFCC features
def extract_features(file_path, max_pad_len=100, num_mfcc=40):
    try:
        audio, sr = sf.read(file_path)
        if len(audio) == 0:
            return np.zeros((num_mfcc, max_pad_len))

        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)  # Convert to mono

        frame_length = int(0.025 * sr)  
        frame_step = int(0.01 * sr)  
        num_frames = max(1, (len(audio) - frame_length) // frame_step + 1)

        if num_frames <= 0:
            return np.zeros((num_mfcc, max_pad_len))

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
    except:
        return np.zeros((num_mfcc, max_pad_len))

# Load dataset
data_path = r"D:\new project"  # Change to your dataset path
labels = []
features = []

for root, _, files in os.walk(data_path):
    for file in files:
        if file.endswith(".wav"):
            parts = file.split("-")  # Split filename by '-'
            if len(parts) > 2:
                emotion_code = parts[2]  # Extract emotion part
                emotion = emotion_map.get(emotion_code, "Unknown")  # Map to label
                if emotion == "Unknown":
                    continue  # Skip unknown labels

                file_path = os.path.join(root, file)
                extracted_feature = extract_features(file_path)
                if extracted_feature is not None and extracted_feature.size > 0:
                    features.append(extracted_feature)
                    labels.append(emotion)

# Convert to numpy arrays
if len(features) == 0:
    raise ValueError("No valid features extracted from dataset. Check the dataset files.")

X = np.array(features)
y = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = to_categorical(y)

# Save the LabelEncoder for testing
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(40, 100)),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(y.shape[1], activation='softmax')
])

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save("speech_emotion_model.h5")
print("Model training complete and saved!")
