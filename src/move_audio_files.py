import os
import shutil

# Paths to your real and fake folders
folders = [
    r"D:\web development\video21\data\real",
    r"D:\web development\video21\data\fake"
]

for parent in folders:
    count = 0
    for root, dirs, files in os.walk(parent):
        for file in files:
            if file.endswith(".wav"):
                src = os.path.join(root, file)
                dst = os.path.join(parent, file)
                if src != dst:  # avoid moving if already in parent
                    shutil.move(src, dst)
                    count += 1
    print(f"✅ Moved {count} .wav files to {parent}")

print("🎉 All .wav files are now at the top level of real/ and fake/ folders.")
