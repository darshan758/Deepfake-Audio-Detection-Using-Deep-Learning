import os
from moviepy.editor import VideoFileClip

# 🔸 Change this path to your actual dataset folder (fake for now)
input_folder = r"D:\web development\video21\data\fake"
output_folder = r"D:\web development\video21\data\fake_wav"

os.makedirs(output_folder, exist_ok=True)

for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith(".mp4") or file.endswith(".wavtolip"):
            video_path = os.path.join(root, file)
            wav_name = os.path.splitext(file)[0] + ".wav"
            wav_path = os.path.join(output_folder, wav_name)
            try:
                clip = VideoFileClip(video_path)
                clip.audio.write_audiofile(wav_path, codec='pcm_s16le')
                clip.close()
                print(f"Extracted: {wav_name}")
            except Exception as e:
                print(f"Error processing {file}: {e}")

print("✅ All audios extracted successfully!")
