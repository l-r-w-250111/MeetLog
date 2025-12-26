import os
import argparse
from dotenv import load_dotenv
import torch
import torch.torch_version
import numpy as np
from pyannote.audio import Pipeline, Inference, Model
from pyannote.audio.core.io import Audio
from pyannote.core import Segment
from pyannote.audio.core.task import Specifications, Problem, Resolution
from pyannote.audio.core.model import Introspection 
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint 
from omegaconf.listconfig import ListConfig
from omegaconf.base import ContainerMetadata, Metadata 
from omegaconf.dictconfig import DictConfig
from omegaconf.nodes import AnyNode
from typing import Any 
from collections import defaultdict
import huggingface_hub
from functools import wraps

# --- Globals for expensive models ---
_diarization_pipeline = None
_embedding_model = None
_device = None

# --- Hugging Face Hub Monkey Patch ---
original_hf_hub_download = huggingface_hub.hf_hub_download
@wraps(original_hf_hub_download)
def _hf_hub_download_wrapper(*args, **kwargs):
    if "use_auth_token" in kwargs:
        kwargs["token"] = kwargs.pop("use_auth_token")
    return original_hf_hub_download(*args, **kwargs)

# パッチの適用
huggingface_hub.hf_hub_download = _hf_hub_download_wrapper
# ------------------------------------


def _initialize_pipelines():
    """Initializes and loads the pyannote pipelines if they haven't been loaded yet."""
    global _diarization_pipeline, _embedding_model, _device
    if _diarization_pipeline is not None:
        return

    # --- PyTorch/CUDA Diagnostic ---
    print("\n--- PyTorch/CUDA Diagnostic (diarize.py) ---")
    print(f"PyTorch Version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print("--------------------------------------------\n")
    # --- End Diagnostic ---

    # [FIX] pyannote.audio/torchaudio compatibility patch for torchaudio > 2.2.0
    # This is necessary because pyannote.audio 3.4.0 expects AudioDecoder in a
    # different location than where it is in modern torchaudio versions.
    try:
        import torchaudio.io
        from torchcodec.decoders import AudioDecoder
        torchaudio.io.AudioDecoder = AudioDecoder
        print("Successfully applied torchcodec compatibility patch.")
    except (ImportError, RuntimeError) as e:
        # If torchcodec is not installed or FFmpeg is missing, this will fail.
        # We print a warning but allow the app to continue, as transcription
        # might still work even if diarization is broken.
        print(f"Warning: Could not apply torchcodec patch. Diarization may fail. Error: {e}")

    # [FIX] PyTorch >= 2.6 の安全なロードのためのカスタムグローバルを追加
    if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
        print("Adding pyannote custom classes to torch's safe globals...")
        torch.serialization.add_safe_globals([
            torch.torch_version.TorchVersion,
            Specifications,
            Problem,
            Resolution, 
            Introspection, 
            EarlyStopping, 
            ModelCheckpoint,
            ListConfig,
            ContainerMetadata,
            Metadata,
            DictConfig,
            AnyNode, 
            Any,
            list,
            defaultdict,
            dict,
            int
        ])

    load_dotenv()
    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    if not hf_token:
        raise ValueError("HUGGING_FACE_TOKEN not found in .env file. Please add it.")

    print("Loading speaker diarization pipeline for the first time...")
    _diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    if _diarization_pipeline is None:
        print("Error: Failed to load the diarization pipeline. Check token or network.")
        # We don't raise here, but let perform_diarization handle the None pipeline
        return

    print("Loading speaker embedding model for the first time...")
    model = Model.from_pretrained(
        "pyannote/embedding",
        use_auth_token=hf_token
    )
    if model is None:
        print("Error: Failed to load the embedding model. Check token or network.")
        # Mark pipeline as unusable
        _diarization_pipeline = None 
        return

    _embedding_model = Inference(model, window="whole")

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _diarization_pipeline.to(_device)
    _embedding_model.to(_device)
    print(f"Pipelines loaded on device: {_device}")


def perform_diarization(audio_path: str) -> tuple[list, dict]:
    """
    Performs speaker diarization and extracts speaker embeddings.
    Models are loaded lazily on the first call.

    Args:
        audio_path: Path to the audio file.

    Returns:
        A tuple containing:
        - A list of speaker segments.
        - - A dictionary mapping speaker labels to their average embedding vectors.
    """
    # 1. Ensure models are loaded
    _initialize_pipelines()

    # 2. Check if the audio file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    # 3. Apply diarization with error handling
    try:
        print(f"Applying diarization to '{os.path.basename(audio_path)}'...")
        if _diarization_pipeline is None:
            raise RuntimeError("Diarization pipeline is not initialized, likely due to a torchcodec/FFmpeg error.")
        
        # Run the pipeline
        result = _diarization_pipeline(audio_path)
        
        # [FIX] Handle different return types from pyannote pipeline.
        # Sometimes it returns a container with an .annotation attribute,
        # and sometimes (e.g., for MP3s) it returns the annotation object directly.
        if hasattr(result, 'annotation'):
            diarization_result = result.annotation
        else:
            diarization_result = result
            
        print("Diarization complete.")
    except Exception as e:
        # This can happen with very short audio files or if the pipeline fails
        print(f"Warning: Diarization failed. Returning empty result. Error: {e}")
        return [], {}

    # 4. Extract speaker embeddings
    print("Extracting speaker embeddings...")
    try:
        audio_loader = Audio(sample_rate=16000, mono=True)
        file_duration = audio_loader.get_duration(audio_path)

        speaker_embeddings = {}
        # Use the annotation object from the container
        for speaker_label in diarization_result.labels():
            speaker_turns = diarization_result.label_timeline(speaker_label)

            embeddings_list = []
            for turn in speaker_turns:
                # Adjust segment boundaries to not exceed file duration
                if turn.end > file_duration:
                    turn = Segment(turn.start, file_duration)

                if turn.duration <= 0:
                    continue

                chunk, sr = audio_loader.crop(audio_path, turn)
                embedding_input = {"waveform": chunk.to(_device), "sample_rate": sr}

                embedding_windows = _embedding_model(embedding_input)
                
                # [FIX] Handle both 1D and 2D arrays robustly
                embedding_data = np.asarray(embedding_windows.data)
                if embedding_data.ndim == 1:
                    embedding = embedding_data.astype(np.float32)
                else:
                    embedding = np.mean(embedding_data, axis=0).astype(np.float32)
                embeddings_list.append(embedding)

            if embeddings_list:
                # This part should be fine as embeddings_list will contain 1D arrays
                speaker_embeddings[speaker_label] = np.mean(embeddings_list, axis=0).astype(np.float32)

        print(f"Extracted embeddings for {len(speaker_embeddings)} speakers.")

    except Exception as e:
        # This is a critical failure, as transcription depends on it.
        raise RuntimeError(f"Failed during speaker embedding extraction: {e}")

    # 5. Process segments from the annotation object
    segments = []
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        segments.append({
            'start': turn.start,
            'end': turn.end,
            'speaker': speaker
        })

    return segments, speaker_embeddings

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform speaker diarization and extract embeddings.")
    parser.add_argument("audio_file", type=str, help="Path to the audio file.")

    args = parser.parse_args()

    try:
        diarization_segments, embeddings = perform_diarization(args.audio_file)
        
        print("\n--- Diarization Result ---")
        if diarization_segments:
            for seg in diarization_segments:
                print(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] Speaker: {seg['speaker']}")
        else:
            print("No speaker segments found.")

        print("\n--- Speaker Embeddings ---")
        if embeddings:
            for speaker, embedding in embeddings.items():
                print(f"Speaker: {speaker}, Embedding Shape: {embedding.shape}")
        else:
            print("No embeddings extracted.")

    except Exception as e:
        print(f"Error: {e}")
