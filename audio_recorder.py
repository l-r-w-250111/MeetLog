import sounddevice as sd
import numpy as np
import scipy.io.wavfile
import io
import threading

# Standard sample rate for audio processing in this project
SAMPLE_RATE = 16000

def list_input_devices() -> dict:
    """
    Lists available audio input devices.

    Returns:
        A dictionary mapping device names to their indices.
    """
    devices = sd.query_devices()
    input_devices = {}
    for i, device in enumerate(devices):
        # Check if the device has input channels
        if device['max_input_channels'] > 0:
            input_devices[f"{i}: {device['name']}"] = i
    return input_devices

class AudioRecorder:
    """
    A class to handle real-time audio recording using sounddevice in a separate
    daemon thread to prevent the main application from hanging on exit.
    """
    def __init__(self, device_index: int, sample_rate: int = SAMPLE_RATE):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.is_recording = False
        self.audio_buffer = []
        self.latest_volume = 0.0
        self.lock = threading.Lock()
        self._processed_chunks_count = 0
        self.stop_event = threading.Event()
        self.thread = None

    def _callback(self, indata: np.ndarray, frames: int, time, status):
        """
        This is called by the sounddevice stream for each new audio chunk.
        It also calculates the volume of the chunk.
        """
        if status:
            print(f"Audio callback status: {status}")
        with self.lock:
            self.audio_buffer.append(indata.copy())
            # Calculate the RMS of the chunk to represent volume
            volume_rms = np.sqrt(np.mean(indata**2))
            self.latest_volume = float(volume_rms)

    def _record_loop(self):
        """The main loop for the recording thread."""
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                device=self.device_index,
                channels=1,
                dtype='float32',
                callback=self._callback
            ) as stream:
                print("Recording thread started.")
                # Wait until the stop event is set
                self.stop_event.wait()
            print("Recording stream finished.")
        except Exception as e:
            print(f"Error in recording thread: {e}")

    def start(self):
        """
        Starts the audio recording in a separate daemon thread.
        """
        if self.is_recording:
            print("Already recording.")
            return

        print(f"Starting recording on device {self.device_index}...")
        self.audio_buffer = []
        self._processed_chunks_count = 0
        self.stop_event.clear()
        
        # [FIX] Run the recording loop in a daemon thread
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()
        
        self.is_recording = True
        print("Recording started.")

    def stop(self):
        """
        Stops the audio recording by signaling the thread to exit.
        """
        if not self.is_recording:
            print("Not currently recording.")
            return

        print("Stopping recording...")
        self.stop_event.set() # Signal the thread to stop
        if self.thread:
            self.thread.join(timeout=2) # Wait for the thread to finish
            if self.thread.is_alive():
                print("Warning: Recording thread did not terminate cleanly.")
            self.thread = None
            
        self.is_recording = False
        print("Recording stopped.")

    def get_latest_volume(self) -> float:
        """
        Returns the volume (RMS) of the most recent audio chunk.
        This can be used for a real-time volume meter.
        """
        with self.lock:
            return self.latest_volume

    def get_wav_data(self) -> bytes | None:
        """
        Returns the entire recorded audio buffer as WAV formatted bytes.
        The lock is held only for the brief moment of copying the buffer list
        to avoid blocking the audio callback for an extended time.

        Returns:
            A bytes object containing the audio in WAV format. Returns None if
            there is no audio data.
        """
        with self.lock:
            # Quickly copy the list of chunks under lock
            if not self.audio_buffer:
                return None
            audio_chunks = list(self.audio_buffer)

        # Perform the potentially slow concatenation outside the lock
        full_audio = np.concatenate(audio_chunks, axis=0)
        audio_int16 = np.int16(full_audio * 32767)

        wav_buffer = io.BytesIO()
        scipy.io.wavfile.write(wav_buffer, self.sample_rate, audio_int16)
        wav_buffer.seek(0)

        return wav_buffer.read()

    def get_new_wav_data(self) -> bytes | None:
        """
        Returns only the new, unprocessed audio chunks as WAV formatted bytes.
        This is designed for lightweight, real-time preview processing.

        Returns:
            A bytes object with the new audio in WAV format, or None if no new
            audio is available.
        """
        new_chunks = []
        with self.lock:
            num_chunks = len(self.audio_buffer)
            if num_chunks > self._processed_chunks_count:
                new_chunks = self.audio_buffer[self._processed_chunks_count:num_chunks]
                self._processed_chunks_count = num_chunks
        
        if not new_chunks:
            return None

        # Process outside the lock
        new_audio = np.concatenate(new_chunks, axis=0)
        audio_int16 = np.int16(new_audio * 32767)
        
        wav_buffer = io.BytesIO()
        scipy.io.wavfile.write(wav_buffer, self.sample_rate, audio_int16)
        wav_buffer.seek(0)
        
        return wav_buffer.read()


if __name__ == "__main__":
    import time

    print("Available input devices:")
    devices = list_input_devices()
    for name, index in devices.items():
        print(f"  - {name}")

    if not devices:
        print("\nNo input devices found. Exiting.")
    else:
        # Example: Use the first available device
        first_device_index = list(devices.values())[0]

        recorder = AudioRecorder(device_index=first_device_index)

        print(f"\nStarting a 5-second test recording on device {first_device_index}...")
        recorder.start()
        time.sleep(2)
        
        print("Getting first chunk of new data...")
        new_data = recorder.get_new_wav_data()
        print(f"Got {len(new_data) if new_data else 0} bytes.")

        time.sleep(3)

        print("Getting second chunk of new data...")
        new_data_2 = recorder.get_new_wav_data()
        print(f"Got {len(new_data_2) if new_data_2 else 0} bytes.")

        recorder.stop()

        print("\nRecording complete.")
        wav_data = recorder.get_wav_data()

        if wav_data:
            # Save the recording to a file for verification
            output_filename = "test_recording.wav"
            with open(output_filename, 'wb') as f:
                f.write(wav_data)
            print(f"Recording saved to '{output_filename}'.")
        else:
            print("No audio was recorded.")