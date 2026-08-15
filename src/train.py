import torch
import os
from pathlib import Path
from torch.utils.data import random_split, DataLoader
from model import SimpleLSTM
from dataset import AudioDeepfakeDataset
from trainer import ModelTrainer
from utils import set_seed_all, set_benchmark_mode

# -----------------------------
# 1. Reproducibility
# -----------------------------
set_seed_all(42)
set_benchmark_mode()

# -----------------------------
# 2. Prepare dataset
# -----------------------------
real_files = list(Path("../data/real").glob("*.wav"))
fake_files = list(Path("../data/fake").glob("*.wav"))

print(f"📊 Dataset Summary -> Real: {len(real_files)}, Fake: {len(fake_files)}")

# ⚖️ Balance dataset
min_count = min(len(real_files), len(fake_files))
if len(real_files) != len(fake_files):
    print(f"⚖️ Balanced Dataset -> Using {min_count} real and {min_count} fake files.")
    real_files = real_files[:min_count]
    fake_files = fake_files[:min_count]

file_list = [str(f) for f in real_files + fake_files]
labels = [0] * len(real_files) + [1] * len(fake_files)  # 0 = real, 1 = fake

if len(file_list) == 0:
    raise ValueError("⚠️ No audio files found in ../data/real or ../data/fake!")

dataset = AudioDeepfakeDataset(file_list, labels, n_mfcc=40)

# Split dataset (80% train, 20% test)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
dataset_train, dataset_test = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=16)

# -----------------------------
# 3. Model Setup
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleLSTM(input_dim=40, hidden_dim=128, num_layers=2).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------
# 4. Checkpoint Handling
# -----------------------------
CHECKPOINT_PATH = "../saved/checkpoint.pth"
start_epoch = 1
total_epochs = 15

if os.path.exists(CHECKPOINT_PATH):
    print("🔁 Found checkpoint! Resuming training...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"➡️ Resuming from epoch {start_epoch}")
else:
    print("🚀 Starting new training session...")

# -----------------------------
# 5. Trainer Setup
# -----------------------------
trainer = ModelTrainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device
)

# -----------------------------
# 6. Training (Single Loop)
# -----------------------------
print(f"🚀 Starting training from epoch {start_epoch} to {total_epochs}...\n")

trainer.train(
    train_loader=train_loader,
    val_loader=test_loader,
    epochs=total_epochs,
    save_dir="../saved"
)

print("✅ Training completed successfully!")
