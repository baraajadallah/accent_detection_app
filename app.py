import streamlit as st
import torch
import torchaudio
import tempfile
import requests
import os
#from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
#import ffmpeg
import subprocess
import yt_dlp
import gdown
import random
import string
from imageio_ffmpeg import get_ffmpeg_exe
import imageio_ffmpeg



# ----------------------------
# 🔧 Configuration
# ----------------------------
MODEL_NAME = "HamzaSidhu786/speech-accent-detection"


# ----------------------------
# 🧠 Load Model
# ----------------------------
@st.cache_resource
def load_model():
    try:
        st.write("🔄 Loading extractor...")
        #processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

        st.write("🔄 Loading model...")
        model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)

        model.eval()
        st.success("✅ Model loaded successfully.")
        return extractor, model

    except Exception as e:
        import traceback
        st.error("❌ Failed to load model.")
        st.code(traceback.format_exc())
        raise


# ----------------------------
# 📥 Download Video from URL
# ----------------------------
def download_video(url: str) -> str:
    try:
        # Handle YouTube or Loom (via yt-dlp)
        if "youtube.com" in url or "youtu.be" in url or "loom.com" in url:
            # Use a unique random filename, let yt-dlp set extension
            temp_dir = tempfile.gettempdir()
            filename = ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))
            outtmpl = os.path.join(temp_dir, f"{filename}.%(ext)s")

            ydl_opts = {
                'outtmpl': outtmpl,
                'format': 'best[ext=mp4]/best',
                'quiet': True,
                'postprocessors': []
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=True)
                real_file = ydl.prepare_filename(info_dict)

                # Handle merged output case
                if not real_file.endswith(".mp4"):
                    real_file = real_file.rsplit(".", 1)[0] + ".mp4"

            if not os.path.exists(real_file) or os.path.getsize(real_file) < 1_000_000:
                raise RuntimeError("Downloaded video is invalid or too small.")
            
            print(f"[yt-dlp] Downloaded video: {real_file}")
            return real_file

        # Handle Google Drive
        elif "drive.google.com" in url:
            tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp_path = tmp_video.name
            file_id = url.split("/d/")[1].split("/")[0]
            direct_url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(direct_url, tmp_path, quiet=True)

            if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) < 1_000_000:
                raise RuntimeError("Google Drive download failed or file too small.")
            
            print(f"[gdown] Downloaded video: {tmp_path}")
            return tmp_path

        # Handle direct links (.mp4, Dropbox, S3, etc.)
        elif url.endswith(".mp4") or "dropbox" in url or "s3.amazonaws.com" in url:
            tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp_path = tmp_video.name

            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(tmp_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

            if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) < 1_000_000:
                raise RuntimeError("Direct download failed or file too small.")
            
            print(f"[requests] Downloaded video: {tmp_path}")
            return tmp_path

        else:
            raise RuntimeError("❌ Unsupported or unrecognized video URL platform.")

    except Exception as e:
        raise RuntimeError(f"Video download failed: {e}")


# ----------------------------
# 🎧 Extract Audio Using FFmpeg
# ----------------------------
def extract_audio(video_path: str, audio_path: str):
    try:
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg_path,
            "-i", video_path,
            "-ac", "1",            # mono channel
            "-ar", "16000",        # sample rate
            "-vn",                 # no video
            audio_path
        ]
        subprocess.run(command, check=True)
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 10000:
            raise RuntimeError("Extracted audio is missing or too small.")
        return audio_path
    except Exception as e:
        raise RuntimeError(f"Audio extraction failed: {e}")


# ----------------------------
# 🔍 Predict Accent from Audio
# ----------------------------
def predict_accent(audio_path: str, extractor, model):
    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)

        waveform = waveform[0].unsqueeze(0)  # Use mono channel

        inputs = extractor(waveform[0], sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_id].item()
            label = model.config.id2label[pred_id]

        explanation = f"Detected **{label}** accent with **{round(confidence * 100, 2)}%** confidence."
        return label, confidence * 100, explanation
    except Exception as e:
        raise RuntimeError(f"Accent prediction failed: {e}")


# ----------------------------
# 🚀 Streamlit Web App
# ----------------------------
st.set_page_config(page_title="English Accent Detector", layout="centered")
st.title("🎙️ English Accent Detector")
st.markdown("Paste a public `.mp4` video URL. This tool will extract audio and classify the English-speaking accent.")

with st.expander("ℹ️ Supported Platforms & Guidelines"):
    st.markdown("""
    - ✅ Video must be **publicly accessible** (no login required).
    - ✅ Supported platforms: **YouTube**, **Loom**, **Google Drive**, **Dropbox**, and direct `.mp4` links.
    - 🎯 Only **English-speaking accents** are supported (e.g., American, British, Indian, Australian).
    - 🔊 Clear, uninterrupted speech works best — avoid background noise, music, or multiple speakers.
    """)

video_url = st.text_input("🔗 Enter Public Video URL (MP4 or Loom):")

if st.button("Analyze"):
    if not video_url.strip():
        st.warning("⚠️ Please enter a valid video URL.")
    else:
        with st.spinner("⏳ Downloading and analyzing video..."):
            try:
                # Load model
                extractor, model = load_model()

                # Temp files
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
                    video_path = download_video(video_url)#, tmp_video.name)
                    audio_path = video_path.replace(".mp4", ".wav")

                    extract_audio(video_path, audio_path)
                    accent, score, explanation = predict_accent(audio_path, extractor, model)

                # Show result
                st.success("✅ Analysis Complete!")
                st.write(f"**Predicted Accent:** {accent}")
                st.write(f"**English Accent Confidence:** {round(score, 2)}%")
                st.info(explanation)

            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
