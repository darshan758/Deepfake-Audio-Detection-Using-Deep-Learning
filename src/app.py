from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchaudio
import os
import torch.nn.functional as F
from model import SimpleLSTM  # your LSTM model class

app = Flask(__name__)
CORS(app)

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = r"D:\web development\video21\saved\checkpoint.pth"
MAX_SEQ_LEN = 200
CHUNK_SEC = 3
SAMPLE_RATE = 16000

model = SimpleLSTM(input_dim=40, hidden_dim=128, num_layers=2).to(device)

# Load model
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print("✅ Model loaded successfully!")
else:
    print("⚠️ Model checkpoint not found!")

# MFCC Transform
mfcc_transform = torchaudio.transforms.MFCC(sample_rate=SAMPLE_RATE, n_mfcc=40)

def preprocess_audio(file_path):
    waveform, sr = torchaudio.load(file_path)
    print(f"Waveform shape: {waveform.shape}, Sample rate: {sr}")

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

    return waveform

def get_chunks(waveform):
    chunk_len = CHUNK_SEC * SAMPLE_RATE
    num_chunks = (waveform.shape[1] + chunk_len - 1) // chunk_len
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_len
        end = start + chunk_len
        chunk = waveform[:, start:end]

        if chunk.shape[1] < chunk_len:
            pad = torch.zeros((1, chunk_len - chunk.shape[1]))
            chunk = torch.cat([chunk, pad], dim=1)

        chunks.append(chunk)
    return chunks

def predict_audio(waveform):
    chunks = get_chunks(waveform)
    probs_list = []

    for chunk in chunks:
        mfcc = mfcc_transform(chunk).squeeze(0).transpose(0, 1)

        if mfcc.shape[0] < MAX_SEQ_LEN:
            pad = torch.zeros(MAX_SEQ_LEN - mfcc.shape[0], mfcc.shape[1])
            mfcc = torch.cat([mfcc, pad], dim=0)
        else:
            mfcc = mfcc[:MAX_SEQ_LEN]

        mfcc = mfcc.unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(mfcc)
            mean_output = output.mean(dim=0)

            if mean_output.dim() == 0:
                mean_output = mean_output.unsqueeze(0)

            prob = F.softmax(mean_output, dim=0)
            probs_list.append(prob.cpu())

    avg_prob = torch.stack(probs_list).mean(dim=0)
    pred_class = torch.argmax(avg_prob).item()
    confidence = avg_prob[pred_class].item() * 100

    label = "REAL" if pred_class == 0 else "FAKE"
    return f"{label} ({confidence:.2f}% confidence)"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    file.save(file_path)

    try:
        waveform = preprocess_audio(file_path)
        result = predict_audio(waveform)

        # 🔥 NEW: Print prediction in backend terminal
        print("🎯 Final Prediction:", result)

        return jsonify({"result": result})

    except Exception as e:
        print("❌ Error:", e)  # print errors too
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "Backend API for Deepfake Audio Detection"

if __name__ == "__main__":
    app.run(debug=True)
