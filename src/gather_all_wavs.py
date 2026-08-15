import os, shutil

base_folders = [
    r"D:\web development\video21\data\fake",
    r"D:\web development\video21\data\real"
]

for base in base_folders:
    count = 0
    for root, dirs, files in os.walk(base):
        for file in files:
            if file.endswith(".wav"):
                src = os.path.join(root, file)
                dst = os.path.join(base, file)
                if src != dst:
                    try:
                        shutil.move(src, dst)
                        count += 1
                    except Exception as e:
                        print("Error:", e)
    print(f"✅ Moved {count} .wav files to {base}")
print("🎉 All .wav files are now in fake/ and real/ top-level folders.")
