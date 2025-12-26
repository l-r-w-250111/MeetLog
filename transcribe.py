from faster_whisper import WhisperModel
import torch
import os
import argparse
import numpy as np

# --- Globals for expensive models ---
_whisper_models = {}

def load_model(model_size: str):
    """Loads a faster-whisper model or returns the cached model."""
    global _whisper_models
    if model_size not in _whisper_models:
        # --- PyTorch/CUDA Diagnostic ---
        print("\n--- PyTorch/CUDA Diagnostic (transcribe.py) ---")
        print(f"PyTorch Version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")
        if cuda_available:
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            print(f"Current GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print("---------------------------------------------\n")
        # --- End Diagnostic ---

        print(f"Loading faster-whisper model '{model_size}' for the first time...")
        device = "cpu"
        compute_type = "int8"
        try:
            if torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"
                print("CUDA is available. Using GPU with float16 compute type.")
            else:
                 print("CUDA not available. Using CPU with int8 compute type.")
            
            _whisper_models[model_size] = WhisperModel(
                model_size, device=device, compute_type=compute_type
            )
        except Exception as e:
            print(f"Warning: Failed to load model on CUDA, falling back to CPU. Error: {e}")
            device = "cpu"
            compute_type = "int8"
            _whisper_models[model_size] = WhisperModel(
                model_size, device=device, compute_type=compute_type
            )

        print("Model loaded.")
    return _whisper_models[model_size]


def transcribe_audio_chunk(audio_source: str | np.ndarray, model_size: str) -> str:
    """
    Transcribes a single audio chunk using the specified faster-whisper model.
    Models are cached globally to avoid reloading.

    Args:
        audio_source (str or np.ndarray): Path to audio file or float32 NumPy array.
        model_size (str): The faster-whisper model size (e.g., 'base', 'small').

    Returns:
        The transcribed text of the audio chunk.
    """
    if isinstance(audio_source, str) and not os.path.exists(audio_source):
        raise FileNotFoundError(f"Audio file not found: {audio_source}")

    model = load_model(model_size)

    try:
        segments, _ = model.transcribe(audio_source)
        # Concatenate segment texts
        text_parts = [segment.text for segment in segments]
        text = "".join(text_parts).strip()
        
        if text: # Avoid printing empty transcriptions
            print(f"Transcribed chunk: {text}")
        return text
    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transcribe an audio file using faster-whisper.")
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
