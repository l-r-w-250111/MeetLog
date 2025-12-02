# 🎙️ Meeting-Minutes-with-AI: AI-Powered Transcription & Diarization Tool

This project provides a powerful tool to automatically generate meeting transcripts from an audio file, complete with speaker identification. It leverages the cutting-edge `openai-whisper` for highly accurate transcription and `pyannote.audio` for state-of-the-art speaker diarization.

Never forget who said what in a meeting again!

---

## ✨ Features

-   **Accurate Transcription**: Utilizes OpenAI's Whisper models to convert speech to text with high precision.
-   **Speaker Diarization**: Identifies different speakers in the audio and labels their speech segments (e.g., `SPEAKER_00`, `SPEAKER_01`).
-   **Speaker Recognition & Profiling**:
    -   **Assign Names**: Give custom names to speaker labels (e.g., "Alice", "Bob").
    -   **Save Profiles**: Save the voice characteristics (embedding) of named speakers to a `speaker_profiles.json` file.
    -   **Automatic Recognition**: Automatically identifies saved speakers in future sessions.
-   **Speaker Selection**: A handy sidebar allows you to choose which saved speaker profiles to use for recognition in any given session.
-   **User-Friendly Web UI**: An intuitive interface built with Streamlit for easy file uploads and interaction.
-   **Command-Line Interface**: A CLI is also available for power users and integration into scripts.

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed on your system:

-   **Python 3.8+**
-   **ffmpeg**: This is essential for audio processing.
    -   **macOS (via Homebrew):** `brew install ffmpeg`
    -   **Ubuntu/Debian:** `sudo apt-get install ffmpeg`
    -   **Windows:**
        1.  Download a build from the [ffmpeg official website](https://ffmpeg.org/download.html) (e.g., from `gyan.dev`).
        2.  Unzip the downloaded file.
        3.  Add the path to the `bin` folder (e.g., `C:\ffmpeg\bin`) to your system's `Path` environment variable.
-   **A Hugging Face Account**: Required to access the pre-trained models from `pyannote.audio`. You can sign up for free at [huggingface.co](https://huggingface.co/).

---

## 🚀 Setup & Installation

Follow these steps to get the project up and running.

### 1. Clone the Repository

```bash
git clone https://github.com/l-r-w-250111/MeetLog.git
cd MeetLog
```

### 2. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# Create the environment
py -3.11 -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Required Libraries

Install all necessary Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```
**(Optional) For NVIDIA GPU Users:** If you have a CUDA-compatible NVIDIA GPU, you can significantly speed up processing. Install the CUDA-enabled version of PyTorch:
```bash
# First, uninstall the default CPU versions
pip uninstall torch torchaudio
# Then, install the CUDA version (example for CUDA 12.1)
pip install torch==2.3.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```
*Find the correct command for your specific CUDA version on the [PyTorch website](https://pytorch.org/get-started/locally/).*

### 4. Set Up Your Hugging Face Access Token 🔑

This is a **critical step** to allow the application to download the necessary speaker diarization models.

#### **Step 4.1: Get Your Token**

-   Log in to your [Hugging Face account](https://huggingface.co/).
-   Go to your account settings and navigate to the **[Access Tokens](https://huggingface.co/settings/tokens)** page.
-   Create a new token with `read` permissions. Copy this token to your clipboard.

#### **Step 4.2: Accept Model User Conditions**

You must accept the user conditions for the following models on the Hugging Face website. **The application will not work if you skip this step!**

1.  Visit [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and accept the terms.
2.  Visit [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and accept the terms.
3.  Visit [pyannote/embedding](https://huggingface.co/pyannote/embedding) and accept the terms.

#### **Step 4.3: Create the `.env` File**

-   In the project's root directory, you will find a file named `.env.example`.
-   **Create a copy** of this file and name it `.env`.
-   Open the new `.env` file and paste your Hugging Face access token into it.

Your `.env` file should look like this:
```
# Replace with your actual Hugging Face access token.
HUGGING_FACE_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```
The application will automatically load this token to authenticate with Hugging Face.

---

## 🏃‍♀️ How to Run

### Using the Web UI (Recommended)

The easiest way to use the tool is through the Streamlit web interface.

1.  **Launch the application:**
    ```bash
    streamlit run app.py
    ```
2.  Your web browser will open with the application's UI.
3.  **Upload** your audio file (`.wav`, `.mp3`, etc.).
4.  **Select** the desired Whisper model size (larger is more accurate but slower).
5.  Click **"Create Transcript"** and wait for the magic to happen!
6.  Once complete, you can **name new speakers**, **save their profiles**, and **download** the final transcript.

### Using the Command-Line Tool

You can also run the tool directly from the command line.

```bash
# Basic usage
python main.py path/to/your/audio.wav

# Use a larger model for better accuracy
python main.py path/to/your/audio.wav --model large
```

---

## 🔧 Troubleshooting

-   **`ffmpeg` Errors**: If you get errors related to reading audio files, it's likely `ffmpeg` is not installed correctly or not in your system's `PATH`. Please revisit the "Prerequisites" section.
-   **GPU Not Used**: If the console shows "Using CPU...", ensure you have an NVIDIA GPU, the latest drivers, and the correct CUDA-enabled version of PyTorch installed.
-   **Hugging Face Errors (401 Unauthorized)**: This usually means your access token is incorrect or you haven't accepted the user conditions for all three `pyannote` models listed in Step 4.2.
-   **`subtype is unknown to TorchAudio` Errors**: Complete silence segments introduced by algorithms will cause an error. Use an editor (e.g., Audacity's Effect > Special > Truncate Silence) to remove them.
---
## 📝 License
This project is licensed under the terms of the [MIT License](LICENSE).
