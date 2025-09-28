import argparse
import whisper
import os
import torch

# Cache for the loaded model to avoid reloading it every time.
_model = None

def transcribe_audio_chunk(audio_path: str, model_size: str = "base") -> str:
    """
    Transcribes an audio file using the local Whisper model.

    Args:
        audio_path: Path to the audio file.
        model_size: The size of the Whisper model to use (e.g., "tiny", "base", "small").

    Returns:
        The transcribed text.
    """
    global _model

    # 1. Check if the audio file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    # 2. Load the Whisper model (or use cached model)
    if _model is None:
        print(f"Loading Whisper model '{model_size}'...")
        # Check for GPU
        if torch.cuda.is_available():
            device = "cuda"
            print("GPU (CUDA) is available. Using GPU for transcription.")
        else:
            device = "cpu"
            print("GPU (CUDA) not found. Using CPU for transcription. This will be significantly slower.")
            print("For GPU acceleration, please ensure you have an NVIDIA GPU and that PyTorch with CUDA support is installed correctly.")
        
        _model = whisper.load_model(model_size, device=device)
        print(f"Model loaded on device: {device}")

    # 3. Perform transcription
    print(f"Transcribing '{os.path.basename(audio_path)}'...")
    # fp16 is only available on CUDA
    result = _model.transcribe(audio_path, fp16=torch.cuda.is_available())
    
    transcribed_text = result["text"]
    print("Transcription complete.")
    
    return transcribed_text.strip()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transcribe an audio file using a local Whisper model.")
    parser.add_argument("audio_file", type=str, help="The path to the audio file to transcribe.")
    parser.add_argument("--model", type=str, default="base", help="The Whisper model size to use (e.g., tiny, base, small, medium, large).")

    args = parser.parse_args()

    try:
        # Example usage: python transcribe.py path/to/your/audio.wav --model base
        transcription = transcribe_audio_chunk(args.audio_file, model_size=args.model)
        
        print("\n--- Transcription Result ---")
        print(transcription)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")