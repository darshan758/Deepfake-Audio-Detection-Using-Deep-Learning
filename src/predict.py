import torch
import torchaudio
from model import SimpleLSTM
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 1. Load the trained model
# -----------------------------
model_path = "../saved/checkpoint.pth"

model = SimpleLSTM(input_dim=40, hidden_dim=128, num_layers=2).to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.eval()
print("✅ Model loaded successfully!")

# -----------------------------
# 2. Predict Function
# -----------------------------
def predict_audio(audio_path):
    waveform, sr = torchaudio.load(audio_path)

    # Convert stereo → mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample → 16 kHz
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

    # Extract MFCC features (same as training)
    transform = torchaudio.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=40,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40},
    )
    mfcc = transform(waveform).squeeze(0).T  # shape: (time, features)

    # Add batch dimension: (1, time, features)
    mfcc = mfcc.unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(mfcc)
        _, predicted = torch.max(output, 1)

    label = "FAKE" if predicted.item() == 1 else "REAL"
    print(f"🎧 Audio File: {os.path.basename(audio_path)}")
    print(f"🧠 Prediction: {label}\n")

# -----------------------------
# 3. Test the function
# -----------------------------
audio_path = "../data/test/sample.wav"  # your .wav file path
predict_audio(audio_path)
