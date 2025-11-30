import os
import argparse
import warnings
import numpy as np

# --- Globals for expensive models ---
_whisper_models = {}
_model_device = None
_compute_type = None

def _initialize_model(model_size: str):
    """Initializes and loads a faster-whisper model, with robust device fallback."""
    global _whisper_models, _model_device, _compute_type

    if model_size in _whisper_models:
        return

    # Lazy import of heavy libraries
    import torch
    from faster_whisper import WhisperModel

    # Determine the device and compute type
    if _model_device is None:
        try:
            if torch.cuda.is_available():
                _model_device = "cuda"
                _compute_type = "float16"
                # Simple check to confirm CUDA is responsive
                _ = torch.tensor([1.0]).to(_model_device)
                print("CUDA is available. Using GPU for Whisper.")
            else:
                _model_device = "cpu"
                _compute_type = "int8" # int8 quantization for CPU is faster
                print("CUDA not available. Using CPU for Whisper.")
        except Exception as e:
            warnings.warn(
                f"CUDA device check failed with error: {e}. "
                f"Falling back to CPU for Whisper model. This may be slow."
            )
            _model_device = "cpu"
            _compute_type = "int8"

    print(f"Loading Whisper model '{model_size}' on '{_model_device}' with compute type '{_compute_type}'...")
    try:
        # Load the model with the determined device and compute type
        _whisper_models[model_size] = WhisperModel(
            model_size, device=_model_device, compute_type=_compute_type
        )
    except Exception as e:
        # If loading on GPU fails, try falling back to CPU
        if _model_device == 'cuda':
            warnings.warn(
                f"Failed to load Whisper model on CUDA with error: {e}. "
                "Attempting to fall back to CPU."
            )
            _model_device = "cpu"
            _compute_type = "int8"
            try:
                _whisper_models[model_size] = WhisperModel(
                    model_size, device=_model_device, compute_type=_compute_type
                )
            except Exception as cpu_e:
                raise RuntimeError(
                    "Failed to load Whisper model on both CUDA and CPU. "
                    f"CUDA error: {e}, CPU error: {cpu_e}"
                )
        else:
            raise RuntimeError(f"Failed to load Whisper model on CPU: {e}")

    print(f"Whisper model '{model_size}' loaded successfully.")

def transcribe_audio_chunk(audio_source: str | np.ndarray, model_size: str) -> str:
    """
    Transcribes a single audio chunk using the specified faster-whisper model.
    Models are cached globally to avoid reloading.

    Args:
        audio_source (str or np.ndarray): Path to audio file or float32 NumPy array.
        model_size (str): The Whisper model size (e.g., 'base', 'small').

    Returns:
        The transcribed text of the audio chunk.
    """
    if isinstance(audio_source, str) and not os.path.exists(audio_source):
        raise FileNotFoundError(f"Audio file not found: {audio_source}")

    # Ensure the model is loaded
    _initialize_model(model_size)
    model = _whisper_models[model_size]

    try:
        # faster-whisper's transcribe method returns an iterator of segments
        segments, _ = model.transcribe(audio_source)

        # Concatenate segments to form the full text
        transcribed_text = " ".join(segment.text for segment in segments).strip()

        if transcribed_text:
            print(f"Transcribed chunk: {transcribed_text}")
        return transcribed_text
    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transcribe an audio file using faster-whisper.")
    parser.add_argument("audio_file", type=str, help="The path to the audio file to transcribe.")
    parser.add_argument("--model", type=str, default="base", help="Whisper model size (e.g., 'base', 'small', 'medium').")

    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"Error: The file '{args.audio_file}' does not exist.")
    else:
        print(f"Transcribing '{args.audio_file}' with the '{args.model}' model...")
        transcribed_text = transcribe_audio_chunk(args.audio_file, model_size=args.model)
        print("\n--- Transcription Result ---")
        print(transcribed_text)
