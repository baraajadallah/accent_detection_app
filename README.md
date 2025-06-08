# English Accent Detection App

This application uses deep learning to detect and analyze English accents in audio recordings.

## 🧠 Model Information

This app uses a fine-tuned Wav2Vec2 model from Hugging Face:
- **Model**: `HamzaSidhu786/speech-accent-detection`
- **Architecture**: Fine-tuned Wav2Vec2 for accent classification
- **Purpose**: Specialized in detecting and classifying common English accents
- **Base Model**: Wav2Vec2, originally developed by Facebook AI for speech recognition

## ✅ Supported Platforms

The app supports publicly accessible video links from:

- **YouTube**
- **Loom**
- **Google Drive**
- **Dropbox**
- **Direct `.mp4` URLs**

## ⚠️ Guidelines and Limitations

- ✅ Video **must be publicly accessible** (no login or token required)
- 🎯 Only **English-speaking accents** are supported (e.g., American, British, Indian, Australian)
- 🔊 Works best with **clear, uninterrupted speech** and **minimal background noise**
- 👤 Assumes **one speaker per video**
- ❌ Currently unsupported: Facebook, Instagram, TikTok, private Dropbox/Drive/S3 links

## System Requirements

### Python Version
- Python 3.11 or higher

### System Dependencies
Before installing the Python packages, you need to install FFmpeg:

On macOS:
```bash
brew install ffmpeg
```

On Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

On Windows:
Download from [FFmpeg official website](https://ffmpeg.org/download.html)

## Installation

1. Create and activate a virtual environment:
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Running the App

To run the application:
```bash
streamlit run app.py
```

The app will open in your default web browser at http://localhost:8501
