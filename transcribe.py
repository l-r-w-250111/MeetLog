import whisper
import torch
import os
import argparse
import numpy as np

# --- Globals for expensive models ---
_whisper_models = {}

def load_model(model_size: str):
    """Loads a model or returns the cached model."""
    global _whisper_models
    if model_size not in _whisper_models:
        print(f"Loading Whisper model '{model_size}' for the first time...")
        _whisper_models[model_size] = whisper.load_model(model_size)
        print("Model loaded.")
    return _whisper_models[model_size]

def transcribe_audio_chunk(audio_source: str | np.ndarray, model_size: str) -> str:
    """
    Transcribes a single audio chunk using the specified Whisper model.
    Models are cached globally to avoid reloading.

    Args:
        audio_source (str or np.ndarray): Path to audio file or float32 NumPy array.
        model_size (str): The Whisper model size (e.g., 'base', 'small').

    Returns:
        The transcribed text of the audio chunk.
    """
    if isinstance(audio_source, str) and not os.path.exists(audio_source):
        raise FileNotFoundError(f"Audio file not found: {audio_source}")

    model = load_model(model_size)

    # Check for CUDA availability for faster processing
    use_fp16 = torch.cuda.is_available()

    try:
        # The 'audio' parameter of transcribe can be a path or a numpy array
        result = model.transcribe(audio_source, fp16=use_fp16)
        text = result['text'].strip()
        if text: # Avoid printing empty transcriptions
            print(f"Transcribed chunk: {text}")
        return text
    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transcribe an audio file using Whisper.")
    parser.add_argument("audio_file", type=str, help="The path to the audio file to transcribe.")
    parser.add_argument("--model", type=str, default="base", help="Whisper model size.")

    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"Error: The file '{args.audio_file}' does not exist.")
    else:
        print(f"Transcribing '{args.audio_file}' with the '{args.model}' model...")
        transcribed_text = transcribe_audio_chunk(args.audio_file, model_size=args.model)
        print("\n--- Transcription Result ---")
        print(transcribed_text)