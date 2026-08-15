import os
from scipy.io import wavfile
import numpy as np
import resampy

folders = ["../data/real", "../data/fake"]

for folder in folders:
    for file_name in os.listdir(folder):
        if file_name.endswith(".wav"):
            file_path = os.path.join(folder, file_name)
            # Read WAV file
            sr, data = wavfile.read(file_path)
            
            # Convert to 16 kHz if needed
            if sr != 16000:
                data = resampy.resample(data.astype(float), sr, 16000)
                data = data.astype(np.int16)
                sr = 16000
            
            # Convert to mono if stereo
            if len(data.shape) > 1:
                data = data.mean(axis=1).astype(np.int16)
            
            # Save fixed file
            base_name = os.path.splitext(file_name)[0]
            new_file_path = os.path.join(folder, f"{base_name}_fixed.wav")
            wavfile.write(new_file_path, sr, data)
            print(f"Converted {file_name} -> {base_name}_fixed.wav")

print("All audio files converted and ready for training!")
