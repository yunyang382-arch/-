import os
import whisper
import shutil
from tqdm import tqdm
import math

# Add ffmpeg to PATH explicitly
# Based on the previous search, ffmpeg is located here in the conda environment
ffmpeg_dir = r"D:\Users\Admin\anaconda3\envs\pytorch\Library\bin"
if os.path.exists(ffmpeg_dir):
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
    print(f"Prepend {ffmpeg_dir} to PATH")

# Print current working directory
print(f"Current working directory: {os.getcwd()}")

# Check if ffmpeg is available
ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path:
    print(f"ffmpeg is installed and found at: {ffmpeg_path}")
else:
    print("ffmpeg is NOT found in PATH. Transcription will likely fail.")

# Path to the recordings folder
#recordings_dir = r"D:\读研项目\05-USB网卡设计\07视频"
recordings_dir = r"C:\Users\Admin\Downloads\20260113"

if not os.path.exists(recordings_dir):
    os.makedirs(recordings_dir)
    print(f"Created directory: {recordings_dir}")
    print("Please put your audio/video files into this folder.")
    exit()

print(f"Scanning for files in: {recordings_dir}")

# Supported extensions (Added more video formats)
supported_extensions = ('.mp3', '.wav', '.flac', '.m4a', '.mp4', '.mkv', '.mov', '.avi', '.webm')
media_files = [f for f in os.listdir(recordings_dir) if f.lower().endswith(supported_extensions)]

def format_timestamp(seconds: float):
    return "{:02}:{:02}:{:02},{:03}".format(
        int(seconds // 3600),
        int(seconds % 3600 // 60),
        int(seconds % 60),
        int((seconds * 1000) % 1000)
    )

def write_srt(segments, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            text = segment["text"].strip()
            f.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")

if not media_files:
    print(f"No media files found in {recordings_dir}. Please add some files.")
else:
    try:
        # Load model
        # You can change 'tiny' to 'base', 'small', 'medium', 'large' for better accuracy but slower speed.
        print("Loading model...")
        model = whisper.load_model("tiny")
        print("Model loaded.")

        # Transcribe
        print("Starting transcription...")
        print("Real-time output will be shown below (verbose=True).")
        
        for media_file in tqdm(media_files, desc="Total Progress"):
            media_path = os.path.join(recordings_dir, media_file)
            print(f"\n\n=== Transcribing {media_file} ===")
            
            # verbose=True shows real-time text segments
            result = model.transcribe(media_path, verbose=True)
            
            # Save the result to a text file
            base_name = os.path.splitext(media_file)[0]
            txt_path = os.path.join(recordings_dir, base_name + ".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(result["text"])
            print(f"\nsaved text to: {txt_path}")

            # Save the result to an SRT file
            srt_path = os.path.join(recordings_dir, base_name + ".srt")
            write_srt(result["segments"], srt_path)
            print(f"saved subtitles to: {srt_path}")
            
            print("===============================")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"An error occurred: {e}")
