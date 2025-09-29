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

class RealtimeTranscriber:
    """
    Manages the real-time transcription process by periodically processing
    audio from an AudioRecorder. This class is responsible for diarization
    and transcription of new audio segments, but not for speaker recognition,
    which is handled by the UI layer.
    """
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.last_processed_end_time = 0.0  # in seconds
        self.transcript = []  # Stores segments with raw speaker labels (e.g., 'SPEAKER_00')
        print("RealtimeTranscriber initialized.")

    def process_audio(self, recorder: AudioRecorder) -> tuple[list, dict]:
        """
        Processes the latest audio from the recorder, transcribes only the new
        segments, and returns the full transcript and speaker embeddings.

        Args:
            recorder (AudioRecorder): The audio recorder instance.

        Returns:
            A tuple containing:
            - The full transcript data with raw speaker labels.
            - A dictionary of all speaker embeddings from the latest diarization run.
        """
        wav_data = recorder.get_wav_data()
        if not wav_data:
            return self.transcript, {}

        # Diarization requires a file path, so we use a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(wav_data)

        speaker_embeddings = {}
        try:
            # 1. Perform diarization on the entire audio recorded so far
            print("\nProcessing audio buffer for diarization...")
            diarization_segments, speaker_embeddings = perform_diarization(tmp_path)
            if not diarization_segments:
                return self.transcript, speaker_embeddings

            # 2. Transcribe only new segments
            audio = AudioSegment.from_wav(tmp_path)

            new_segments_to_transcribe = [
                seg for seg in diarization_segments if seg['end'] > self.last_processed_end_time
            ]

            if not new_segments_to_transcribe:
                return self.transcript, speaker_embeddings

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

                    # Append with the raw speaker label; recognition is handled by the caller
                    self.transcript.append({
                        "start": f"{start_s:.2f}",
                        "end": f"{end_s:.2f}",
                        "speaker": segment['speaker'], # Raw label (e.g., 'SPEAKER_00')
                        "text": transcribed_text
                    })
                except Exception as e:
                    print(f"Could not transcribe chunk {i}: {e}")

            shutil.rmtree(temp_chunk_dir)

            # Update the processed time and sort the transcript
            self.last_processed_end_time = diarization_segments[-1]['end']
            self.transcript.sort(key=lambda x: float(x['start']))

        except Exception as e:
            print(f"Error during real-time processing: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return self.transcript, speaker_embeddings

if __name__ == '__main__':
    # This test script needs to be updated to reflect the new return signature
    # and the shift of speaker recognition logic to the caller.
    from audio_recorder import list_input_devices
    from speaker_manager import load_profiles
    from main import recognize_speakers

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
        # 1. Get raw transcript and embeddings from the transcriber
        raw_transcript, embeddings = transcriber.process_audio(recorder)

        # 2. Perform speaker recognition (simulating the UI's role)
        print("\nPerforming speaker recognition...")
        active_profiles = {
            name: data for name, data in load_profiles().items()
            if data.get('is_active', True)
        }
        label_map, unrecognized = recognize_speakers(embeddings, active_profiles)

        print("\n--- Final Mapped Transcript ---")
        if raw_transcript:
            for entry in raw_transcript:
                speaker_name = label_map.get(entry['speaker'], entry['speaker'])
                print(f"[{entry['start']}s - {entry['end']}s] {speaker_name}: {entry['text']}")
        else:
            print("No transcript was generated.")

        if unrecognized:
            print("\n--- Unrecognized Speakers ---")
            for label in unrecognized.keys():
                print(f"- {label}")