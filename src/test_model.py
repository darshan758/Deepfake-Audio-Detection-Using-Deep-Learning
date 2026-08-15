import torch
import torchaudio
import os
from model import SimpleLSTM

# ----------------- Config -----------------
MODEL_PATH = "../saved/checkpoint.pth"
AUDIO_FILE = "test.wav"  # replace with your WAV file
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LEN = 200  # must match training

# MFCC parameters (must match training)
SAMPLE_RATE = 16000
N_MFCC = 40
# optional melkwargs if used during training
MEL_KWARGS = {"n_fft": 400, "hop_length": 160, "n_mels": 40}

# ----------------- Load Model -----------------
model = SimpleLSTM(input_dim=N_MFCC, hidden_dim=128, num_layers=2).to(DEVICE)
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print("✅ Model loaded successfully!")
else:
    raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")

# ----------------- Load Audio -----------------
waveform, sr = torchaudio.load(AUDIO_FILE)
print("Waveform shape:", waveform.shape, "Sample rate:", sr)

# Resample if needed
if sr != SAMPLE_RATE:
    resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
    waveform = resample(waveform)

# Convert to mono
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# ----------------- Compute MFCC -----------------
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC, melkwargs=MEL_KWARGS
)
mfcc = mfcc_transform(waveform)  # (1, n_mfcc, time)
mfcc = mfcc.squeeze(0).transpose(0, 1)  # (time, features)

# Pad or truncate to MAX_SEQ_LEN
seq_len = mfcc.shape[0]
if seq_len < MAX_SEQ_LEN:
    pad = torch.zeros(MAX_SEQ_LEN - seq_len, mfcc.shape[1])
    mfcc = torch.cat([mfcc, pad], dim=0)
else:
    mfcc = mfcc[:MAX_SEQ_LEN, :]

mfcc = mfcc.unsqueeze(1).to(DEVICE)  # (seq_len, batch=1, features)
print("MFCC shape after padding/truncating:", mfcc.shape)

# ----------------- Run Model -----------------
with torch.no_grad():
    outputs = model(mfcc)                # (seq_len, batch, num_classes)
    mean_output = outputs.mean(dim=0)[0]  # average over timesteps

    print("Raw outputs (logits or probabilities):", mean_output.cpu().numpy())
    predicted_class = torch.argmax(mean_output).item()
    result = "REAL" if predicted_class == 0 else "FAKE"
    print("Predicted class:", result)
