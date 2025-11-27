import os
import argparse
from dotenv import load_dotenv
import torch
import numpy as np
# Updated imports to include Model
from pyannote.audio import Pipeline, Model, Inference
from pyannote.audio.core.io import Audio
from pyannote.core import Segment

# --- Globals for expensive models ---
_diarization_pipeline = None
_embedding_model = None
_device = None

def _initialize_pipelines():
    """Initializes and loads the pyannote pipelines if they haven't been loaded yet."""
    global _diarization_pipeline, _embedding_model, _device
    
    if _diarization_pipeline is not None:
        return

    load_dotenv()
    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    if not hf_token:
        raise ValueError("HUGGING_FACE_TOKEN not found in .env file. Please add it.")

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {_device}")

    print("Loading speaker diarization pipeline for the first time...")
    # Use 'token' instead of 'use_auth_token'
    _diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token
    )
    _diarization_pipeline.to(_device)
    
    print("Loading speaker embedding model for the first time...")
    # New way to load embedding model for pyannote.audio 3.x/4.x
    embedding_model_obj = Model.from_pretrained(
        "pyannote/embedding", 
        token=hf_token
    )
    # The Inference class now takes a Model object
    _embedding_model = Inference(embedding_model_obj, window="whole")
    _embedding_model.to(_device)

    print("Pipelines loaded.")

def perform_diarization(audio_path: str) -> tuple[list, dict]:
    """
    Performs speaker diarization and extracts speaker embeddings.
    Models are loaded lazily on the first call.

    Args:
        audio_path: Path to the audio file.

    Returns:
        A tuple containing:
        - A list of speaker segments.
        - A dictionary mapping speaker labels to their average embedding vectors.
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
                # Boundary check from original code
                if turn.end > file_duration:
                    turn = Segment(turn.start, file_duration)
                
                if turn.duration <= 0:
                    continue

                chunk, sr = audio_loader.crop(audio_path, turn)
                
                embedding_input = {"waveform": chunk, "sample_rate": sr}
                
                # The Inference object returns a numpy array directly
                embedding = _embedding_model(embedding_input)
                embeddings_list.append(embedding)
            
            if embeddings_list:
                # Stack and average the embeddings for the current speaker
                speaker_embeddings[speaker_label] = np.mean(np.vstack(embeddings_list), axis=0)
        
        print(f"Extracted embeddings for {len(speaker_embeddings)} speakers.")

    except Exception as e:
        raise RuntimeError(f"Failed during speaker embedding extraction for '{audio_path}': {e}")

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
