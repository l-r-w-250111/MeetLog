import os
import argparse
from dotenv import load_dotenv
import torch
import numpy as np
from pyannote.audio import Pipeline, Inference
from pyannote.audio.core.io import Audio
from pyannote.core import Segment

def perform_diarization(audio_path: str) -> tuple[list, dict]:
    """
    Performs speaker diarization and extracts speaker embeddings.

    Args:
        audio_path: Path to the audio file.

    Returns:
        A tuple containing:
        - A list of speaker segments.
        - A dictionary mapping speaker labels to their average embedding vectors.
    """
    # 1. Load environment variables and get Hugging Face token
    load_dotenv()
    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    if not hf_token:
        raise ValueError("HUGGING_FACE_TOKEN not found in .env file. Please add it.")

    # 2. Check if the audio file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    # 3. Initialize pipelines
    print("Loading speaker diarization pipeline...")
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    
    print("Loading speaker embedding model...")
    embedding_model = Inference(
        "pyannote/embedding", 
        use_auth_token=hf_token
    )

    # 4. Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diarization_pipeline.to(device)
    embedding_model.to(device)
    print(f"Pipelines loaded on device: {device}")

    # 5. Apply diarization
    print(f"Applying diarization to '{os.path.basename(audio_path)}'...")
    diarization_result = diarization_pipeline(audio_path)
    print("Diarization complete.")

    # 6. Extract speaker embeddings
    print("Extracting speaker embeddings...")
    try:
        audio_loader = Audio(sample_rate=16000, mono=True)
        file_duration = audio_loader.get_duration(audio_path)
        
        speaker_embeddings = {}
        for speaker_label in diarization_result.labels():
            speaker_turns = diarization_result.label_timeline(speaker_label)
            
            embeddings_list = []
            for turn in speaker_turns:
                # Ensure the segment does not exceed the file duration
                if turn.end > file_duration:
                    turn = Segment(turn.start, file_duration)
                
                # Skip zero-duration segments that might result from the adjustment
                if turn.duration <= 0:
                    continue

                chunk, sr = audio_loader.crop(audio_path, turn)
                
                embedding_input = {"waveform": chunk.to(device), "sample_rate": sr}
                
                embedding_windows = embedding_model(embedding_input)
                embedding = np.mean(embedding_windows.data, axis=0)
                embeddings_list.append(embedding)
            
            if embeddings_list:
                speaker_embeddings[speaker_label] = np.mean(embeddings_list, axis=0)
        
        print(f"Extracted embeddings for {len(speaker_embeddings)} speakers.")

    except Exception as e:
        raise RuntimeError(f"Failed during speaker embedding extraction: {e}")

    # 7. Process segments
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