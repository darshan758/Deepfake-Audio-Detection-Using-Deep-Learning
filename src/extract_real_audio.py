import os
from moviepy.editor import VideoFileClip

# Path to your real video folder
real_video_dir = r"D:\web development\video21\data\real"
output_dir = os.path.join(real_video_dir, "real_wav")
os.makedirs(output_dir, exist_ok=True)

count = 0
for root, dirs, files in os.walk(real_video_dir):
    for file in files:
        if file.lower().endswith(".mp4"):
            video_path = os.path.join(root, file)
            wav_name = os.path.splitext(file)[0] + ".wav"
            audio_path = os.path.join(output_dir, wav_name)

            try:
                clip = VideoFileClip(video_path)
                clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
                clip.close()
                count += 1
                print(f"🎧 Extracted: {wav_name}")
            except Exception as e:
                print(f"⚠️ Skipped {file} - {e}")

print(f"\n✅ All real audios extracted successfully! Total: {count}")
