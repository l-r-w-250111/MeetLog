import os
import tempfile
import time
import shutil
import io
from pydub import AudioSegment
import numpy as np

# Import from existing modules
from audio_recorder import AudioRecorder
from diarize import perform_diarization
from transcribe import transcribe_audio_chunk
from main import create_transcript

class RealtimeTranscriber:
    """
    Manages the transcription process. It now supports two modes:
    1. `process_preview_audio`: A lightweight, fast method for live previews.
    2. `process_final_audio`: A comprehensive method for final, diarized transcripts.
    """
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.preview_transcript = [] # Stores simple text strings for the live preview
        print("RealtimeTranscriber initialized.")

    def _audio_segment_to_numpy(self, audio_segment: AudioSegment) -> np.ndarray:
        """Converts a Pydub AudioSegment to a float32 NumPy array for Whisper."""
        samples = np.array(audio_segment.get_array_of_samples())
        # Normalize to float32
        return samples.astype(np.float32) / np.iinfo(samples.dtype).max

    def process_preview_audio(self, recorder: AudioRecorder) -> list[str]:
        """
        Processes only the newest audio segment for a lightweight live preview.
        This does NOT perform speaker diarization.

        Args:
            recorder (AudioRecorder): The audio recorder instance.

        Returns:
            A list of transcribed text segments from the new audio.
        """
        new_wav_data = recorder.get_new_wav_data()
        if not new_wav_data:
            return self.preview_transcript

        try:
            # Convert the new wav data bytes to a numpy array for transcription
            audio_segment = AudioSegment.from_wav(io.BytesIO(new_wav_data))
            audio_np = self._audio_segment_to_numpy(audio_segment)
            
            # Transcribe the new chunk
            transcribed_text = transcribe_audio_chunk(audio_np, model_size=self.model_size)
            
            if transcribed_text:
                self.preview_transcript.append(transcribed_text)

        except Exception as e:
            print(f"Could not transcribe preview chunk: {e}")
        
        return self.preview_transcript

    def process_final_audio(self, recorder: AudioRecorder) -> tuple[list, dict, dict]:
        """
        Performs a full, high-quality transcription and diarization on the
        entire recorded audio. This is slow and should only be called once
        at the end of a recording session.

        Args:
            recorder (AudioRecorder): The audio recorder instance.

        Returns:
            A tuple containing:
            - The final structured transcript data.
            - A dictionary of unrecognized speakers.
            - A dictionary mapping speaker labels to their display names.
        """
        full_wav_data = recorder.get_wav_data()
        if not full_wav_data:
            return [], {}, {}

        # Diarization requires a file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(full_wav_data)
        
        transcript, unrecognized, label_map = [], {}, {}
        try:
            # Use the main `create_transcript` function for the final processing
            # This function handles both diarization and transcription.
            transcript, unrecognized, label_map = create_transcript(tmp_path, self.model_size)
        except Exception as e:
            print(f"Error during final audio processing: {e}")
            # Re-raise to be caught by the UI
            raise e
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        return transcript, unrecognized, label_map

    def clear_preview_transcript(self):
        """Clears the preview transcript for a new session."""
        self.preview_transcript = []