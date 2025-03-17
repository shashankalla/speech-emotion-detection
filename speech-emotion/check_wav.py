import os
import soundfile as sf

data_path = r"D:\new project"  # Change this to your dataset folder

for root, _, files in os.walk(data_path):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            try:
                audio, sr = sf.read(file_path)
                print(f"✅ Successfully loaded: {file_path}, Sample Rate: {sr}, Length: {len(audio)}")
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")
