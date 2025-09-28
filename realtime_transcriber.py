import os
import tempfile
import time
import shutil
from pydub import AudioSegment
import numpy as np

# Import from existing modules
from audio_recorder import AudioRecorder
from diarize import perform_diarization
from transcribe import transcribe_audio_chunk
from speaker_manager import load_profiles
from main import recognize_speakers # Use the refactored function

class RealtimeTranscriber:
    """
    Manages the real-time transcription process by periodically processing
    audio from an AudioRecorder.
    """
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.last_processed_end_time = 0.0  # in seconds
        self.transcript = []
        self.speaker_map = {}  # Maps temp labels (SPEAKER_00) to persistent names
        self.unrecognized_embeddings = {} # Stores embeddings of new speakers found in this session
        self.known_profiles = self._load_active_profiles()
        print("RealtimeTranscriber initialized.")
        print(f"Loaded {len(self.known_profiles)} active profiles.")

    def _load_active_profiles(self):
        """Loads active speaker profiles."""
        all_profiles = load_profiles()
        return {
            name: data for name, data in all_profiles.items() if data.get('is_active', True)
        }

    def _update_speaker_mapping(self, new_embeddings: dict):
        """
        Matches new speaker embeddings with known profiles and previously
        unrecognized speakers to maintain consistency.
        """
        # Create a combined pool of speakers to match against: saved profiles + speakers found in this session
        profiles_to_match = self.known_profiles.copy()
        for name, embedding in self.unrecognized_embeddings.items():
            profiles_to_match[name] = {'embedding': embedding}

        # Filter out embeddings for speakers we have already identified
        unmapped_embeddings = {
            label: emb for label, emb in new_embeddings.items() if label not in self.speaker_map
        }

        if not unmapped_embeddings:
            return

        # Use the centralized function to get matches and new unrecognized speakers
        label_map, newly_unrecognized = recognize_speakers(unmapped_embeddings, profiles_to_match)

        # Update our session's speaker map with the results
        for temp_label, final_name in label_map.items():
            # If the name is one of the temporary labels, it means it's a new speaker
            if final_name in newly_unrecognized:
                session_speaker_name = f"Speaker {len(self.unrecognized_embeddings) + 1}"
                self.speaker_map[temp_label] = session_speaker_name
                # Add to our session's list of unrecognized speakers for future checks
                self.unrecognized_embeddings[session_speaker_name] = newly_unrecognized[final_name]
                print(f"Identified new session speaker: {session_speaker_name} (was {temp_label})")
            else:
                # The speaker was matched to an existing profile
                self.speaker_map[temp_label] = final_name
                print(f"Matched {temp_label} to existing speaker: {final_name}")


    def process_audio(self, recorder: AudioRecorder) -> list:
        """
        Processes the latest audio from the recorder, updates the transcript,
        and returns the full transcript.
        """
        wav_data = recorder.get_wav_data()
        if not wav_data:
            return self.transcript

        # Diarization requires a file path, so we use a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(wav_data)

        try:
            # 1. Perform diarization on the entire audio recorded so far
            print("\nProcessing audio buffer for diarization...")
            diarization_segments, speaker_embeddings = perform_diarization(tmp_path)
            if not diarization_segments:
                print("No segments found in current audio buffer.")
                return self.transcript

            # 2. Update speaker mapping with any new speakers found
            self._update_speaker_mapping(speaker_embeddings)

            # 3. Transcribe only new segments
            audio = AudioSegment.from_wav(tmp_path)

            new_segments_to_transcribe = [
                seg for seg in diarization_segments if seg['end'] > self.last_processed_end_time
            ]

            if not new_segments_to_transcribe:
                print("No new segments to transcribe.")
                return self.transcript

            print(f"Found {len(new_segments_to_transcribe)} new segments to transcribe.")

            temp_chunk_dir = "temp_realtime_chunks"
            if os.path.exists(temp_chunk_dir):
                shutil.rmtree(temp_chunk_dir)
            os.makedirs(temp_chunk_dir)

            for i, segment in enumerate(new_segments_to_transcribe):
                start_s = segment['start']
                end_s = segment['end']

                # We only want to process the *new* part of a potentially overlapping segment
                if start_s < self.last_processed_end_time:
                    start_s = self.last_processed_end_time

                if start_s >= end_s:
                    continue

                audio_chunk = audio[start_s * 1000 : end_s * 1000]
                chunk_filename = os.path.join(temp_chunk_dir, f"chunk_{i}.wav")
                audio_chunk.export(chunk_filename, format="wav")

                try:
                    transcribed_text = transcribe_audio_chunk(chunk_filename, model_size=self.model_size)
                    speaker_name = self.speaker_map.get(segment['speaker'], segment['speaker'])

                    self.transcript.append({
                        "start": f"{start_s:.2f}",
                        "end": f"{end_s:.2f}",
                        "speaker": speaker_name,
                        "text": transcribed_text
                    })
                except Exception as e:
                    print(f"Could not transcribe chunk {i}: {e}")

            shutil.rmtree(temp_chunk_dir)

            self.last_processed_end_time = diarization_segments[-1]['end']
            self.transcript.sort(key=lambda x: float(x['start']))

        except Exception as e:
            print(f"Error during real-time processing: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return self.transcript

if __name__ == '__main__':
    from audio_recorder import list_input_devices

    print("This is a test script for the RealtimeTranscriber.")

    devices = list_input_devices()
    if not devices:
        print("No input devices found. Cannot run test.")
    else:
        dev_index = list(devices.values())[0]
        print(f"\nUsing device: {list(devices.keys())[0]}")

        recorder = AudioRecorder(device_index=dev_index)
        transcriber = RealtimeTranscriber(model_size="tiny")

        print("\nStarting 10-second recording for test...")
        recorder.start()
        time.sleep(10)
        recorder.stop()
        print("Recording stopped.")

        print("\nProcessing audio...")
        final_transcript = transcriber.process_audio(recorder)

        print("\n--- Final Transcript ---")
        if final_transcript:
            for entry in final_transcript:
                print(f"[{entry['start']}s - {entry['end']}s] {entry['speaker']}: {entry['text']}")
        else:
            print("No transcript was generated.")