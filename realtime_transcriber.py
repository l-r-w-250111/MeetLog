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

    def _audio_segment_to_numpy(self, audio_segment: AudioSegment) -> np.ndarray:
        """Converts a Pydub AudioSegment to a float32 NumPy array for Whisper."""
        samples = np.array(audio_segment.get_array_of_samples())
        # Normalize to float32
        return samples.astype(np.float32) / np.iinfo(samples.dtype).max

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

        # Diarization still requires a file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(wav_data)

        speaker_embeddings = {}
        try:
            print("\nProcessing audio buffer for diarization...")
            diarization_segments, speaker_embeddings = perform_diarization(tmp_path)
            if not diarization_segments:
                return self.transcript, speaker_embeddings

            audio = AudioSegment.from_wav(tmp_path)

            new_segments_to_transcribe = [
                seg for seg in diarization_segments if seg['end'] > self.last_processed_end_time
            ]

            if not new_segments_to_transcribe:
                return self.transcript, speaker_embeddings

            print(f"Found {len(new_segments_to_transcribe)} new segments to transcribe.")

            for i, segment in enumerate(new_segments_to_transcribe):
                start_s = segment['start']
                end_s = segment['end']

                if start_s < self.last_processed_end_time:
                    start_s = self.last_processed_end_time

                if start_s >= end_s:
                    continue

                audio_chunk_segment = audio[start_s * 1000 : end_s * 1000]

                # Convert audio chunk to numpy array for in-memory processing
                audio_np = self._audio_segment_to_numpy(audio_chunk_segment)

                try:
                    # Transcribe directly from the numpy array
                    transcribed_text = transcribe_audio_chunk(audio_np, model_size=self.model_size)

                    self.transcript.append({
                        "start": f"{start_s:.2f}",
                        "end": f"{end_s:.2f}",
                        "speaker": segment['speaker'],
                        "text": transcribed_text
                    })
                except Exception as e:
                    print(f"Could not transcribe chunk {i}: {e}")

            self.last_processed_end_time = diarization_segments[-1]['end']
            self.transcript.sort(key=lambda x: float(x['start']))

        except Exception as e:
            print(f"Error during real-time processing: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return self.transcript, speaker_embeddings

if __name__ == '__main__':
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
        raw_transcript, embeddings = transcriber.process_audio(recorder)

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