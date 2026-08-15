import os
import shutil

# Paths
source_path = r"D:\web development\video21\data\FakeAVCeleb_v1.2"
target_real = r"D:\web development\video21\data\real"
target_fake = r"D:\web development\video21\data\fake"

# Create folders if not exist
os.makedirs(target_real, exist_ok=True)
os.makedirs(target_fake, exist_ok=True)

# Define mapping: which folders count as real/fake
fake_folders = ["FakeVideo-FakeAudio", "RealVideo-FakeAudio"]
real_folders = ["FakeVideo-RealAudio", "RealVideo-RealAudio"]

def copy_wav_files(src_folder, dest_folder):
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            if file.lower().endswith(".wav"):
                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_folder, file)
                shutil.copy2(src_file, dest_file)
                print(f"Copied: {src_file} -> {dest_folder}")

print("Copying FAKE audio files...")
for folder in fake_folders:
    copy_wav_files(os.path.join(source_path, folder), target_fake)

print("\nCopying REAL audio files...")
for folder in real_folders:
    copy_wav_files(os.path.join(source_path, folder), target_real)

print("\n✅ All .wav files have been organized successfully!")
