import os
import argparse
import shutil
from pydub import AudioSegment

# Import functions from our custom modules
from diarize import perform_diarization
from transcribe import transcribe_audio_chunk
from speaker_manager import load_profiles, find_matching_speaker

def create_transcript(audio_path: str, model_size: str = "base") -> tuple[list, dict, dict]:
    """
    Creates a transcript, recognizing known speakers and identifying new ones.

    Args:
        audio_path (str): Path to the input audio file.
        model_size (str): The Whisper model size for transcription.

    Returns:
        A tuple containing:
        - The structured transcript data.
        - A dictionary of unrecognized speakers and their embeddings.
        - A dictionary mapping speaker labels to their display names.
    """
    # 1. Perform speaker diarization and get embeddings
    try:
        diarization_segments, speaker_embeddings = perform_diarization(audio_path)
    except Exception as e:
        print(f"Error during diarization: {e}")
        raise e

    if not diarization_segments:
        print("No speaker segments were identified. Cannot proceed.")
        return None, {}, {}

    # 2. Recognize known speakers
    print("\nAttempting to recognize speakers from saved profiles...")
    all_profiles = load_profiles()
    # Filter for active profiles only
    active_profiles = {
        name: data for name, data in all_profiles.items() if data.get('is_active', True)
    }
    print(f"Found {len(active_profiles)} active speaker profiles to use for recognition.")
    
    speaker_label_map = {}
    unrecognized_speakers = {}

    for label, embedding in speaker_embeddings.items():
        # Match against active profiles only
        matched_name = find_matching_speaker(embedding, active_profiles)
        if matched_name:
            speaker_label_map[label] = matched_name
            print(f"Matched {label} to known speaker: {matched_name}")
        else:
            speaker_label_map[label] = label  # Keep original label for now
            unrecognized_speakers[label] = embedding
            print(f"Could not recognize {label}. Marked as new speaker.")

    # 3. Prepare for audio processing
    print(f"\nLoading audio file: {audio_path}")
    try:
        audio = AudioSegment.from_file(audio_path)
    except Exception as e:
        print(f"Error loading audio file with pydub: {e}")
        raise e
        
    temp_dir = "temp_audio_chunks"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    # 4. Process each segment
    full_transcript = []
    print("\nStarting transcription for each speaker segment...")
    for i, segment in enumerate(diarization_segments):
        start_time = segment['start'] * 1000
        end_time = segment['end'] * 1000
        original_speaker_label = segment['speaker']
        
        speaker_display_name = speaker_label_map.get(original_speaker_label, original_speaker_label)

        print(f"Processing segment {i+1}/{len(diarization_segments)}: Speaker {speaker_display_name} ({original_speaker_label}) [{start_time/1000:.2f}s - {end_time/1000:.2f}s]")

        audio_chunk = audio[start_time:end_time]
        chunk_filename = os.path.join(temp_dir, f"chunk_{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")

        try:
            transcribed_text = transcribe_audio_chunk(chunk_filename, model_size=model_size)
            # Store the original label in the transcript data for consistent mapping
            full_transcript.append({
                "start": f"{start_time/1000:.2f}",
                "end": f"{end_time/1000:.2f}",
                "speaker": original_speaker_label,
                "text": transcribed_text
            })
        except Exception as e:
            print(f"Could not transcribe chunk {i}: {e}")

    # 5. Clean up
    shutil.rmtree(temp_dir)
    print("\nCleaned up temporary files.")

    # 6. Return results
    return full_transcript, unrecognized_speakers, speaker_label_map


def save_transcript_to_file(transcript: list, output_filename: str, speaker_name_map: dict):
    """Saves the structured transcript data to a text file."""
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("--- Meeting Transcript ---\n\n")
        for entry in transcript:
            speaker_display = speaker_name_map.get(entry['speaker'], entry['speaker'])
            line = f"[{entry['start']}s - {entry['end']}s] {speaker_display}: {entry['text']}\n"
            f.write(line)
    print(f"\nTranscript successfully saved to '{output_filename}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a transcript with speaker diarization.")
    parser.add_argument("audio_file", type=str, help="The path to the audio file.")
    parser.add_argument("--model", type=str, default="base", help="Whisper model size.")
    parser.add_argument("--output_file", type=str, default="transcript.txt", help="Output file name.")
    
    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"Error: The file '{args.audio_file}' does not exist.")
    else:
        transcript_data, unrecognized, label_map = create_transcript(args.audio_file, args.model)
        
        if transcript_data:
            save_transcript_to_file(transcript_data, args.output_file, label_map)
            if unrecognized:
                print("\n--- Unrecognized Speakers ---")
                print("The following speakers were not found in profiles:")
                for label in unrecognized.keys():
                    print(f"- {label}")
                print("\nUse the Streamlit app to assign names to these speakers.")
        else:
            print("Transcript generation failed.")