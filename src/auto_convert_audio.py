import os
from pydub import AudioSegment

# Folders
folders = ["../data/real", "../data/fake"]

# Loop through each folder
for folder in folders:
    for file_name in os.listdir(folder):
        if file_name.endswith(".wav"):
            file_path = os.path.join(folder, file_name)
            # Load audio
            audio = AudioSegment.from_file(file_path)
            # Convert to mono, 16kHz, 16-bit PCM
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            # Save as fixed file
            base_name = os.path.splitext(file_name)[0]
            new_file_path = os.path.join(folder, f"{base_name}_fixed.wav")
            audio.export(new_file_path, format="wav")
            print(f"Converted {file_name} -> {base_name}_fixed.wav")

print("All audio files converted and ready for training!")
