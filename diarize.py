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
    print("Loading speaker embedding model for the first time...")
    model = Model.from_pretrained(
        "pyannote/embedding",
        use_auth_token=hf_token
    )
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

    # 3. Apply diarization
    print(f"Applying diarization to '{os.path.basename(audio_path)}'...")
    diarization_result = _diarization_pipeline(audio_path)
    print("Diarization complete.")

    # 4. Extract speaker embeddings
    print("Extracting speaker embeddings...")
    try:
        audio_loader = Audio(sample_rate=16000, mono=True)
        file_duration = audio_loader.get_duration(audio_path)
        
        speaker_embeddings = {}
        for speaker_label in diarization_result.labels():
            speaker_turns = diarization_result.label_timeline(speaker_label)
            
            embeddings_list = []
            for turn in speaker_turns:
                # セグメントの境界がファイルの長さを超えないように調整
                if turn.end > file_duration:
                    turn = Segment(turn.start, file_duration)
                
                if turn.duration <= 0:
                    continue

                chunk, sr = audio_loader.crop(audio_path, turn)
                
                embedding_input = {"waveform": chunk.to(_device), "sample_rate": sr}
                
                embedding_windows = _embedding_model(embedding_input)
                embedding = np.mean(embedding_windows.data, axis=0)
                embeddings_list.append(embedding)
            
            if embeddings_list:
                speaker_embeddings[speaker_label] = np.mean(embeddings_list, axis=0)
        
        print(f"Extracted embeddings for {len(speaker_embeddings)} speakers.")

    except Exception as e:
        raise RuntimeError(f"Failed during speaker embedding extraction: {e}")

    # 5. Process segments
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
