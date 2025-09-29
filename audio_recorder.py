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
    A class to handle real-time audio recording using sounddevice.
    """
    def __init__(self, device_index: int, sample_rate: int = SAMPLE_RATE):
        """
        Initializes the recorder.

        Args:
            device_index (int): The index of the audio device to use.
            sample_rate (int): The sample rate for the recording.
        """
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.is_recording = False
        self.stream = None
        self.audio_buffer = []
        self.latest_volume = 0.0  # To store the volume of the most recent chunk
        self.lock = threading.Lock()

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

    def start(self):
        """
        Starts the audio recording.
        """
        if self.is_recording:
            print("Already recording.")
            return

        print(f"Starting recording on device {self.device_index}...")
        self.audio_buffer = []
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            device=self.device_index,
            channels=1,
            dtype='float32',
            callback=self._callback
        )
        self.stream.start()
        self.is_recording = True
        print("Recording started.")

    def stop(self):
        """
        Stops the audio recording.
        """
        if not self.is_recording:
            print("Not currently recording.")
            return

        print("Stopping recording...")
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_recording = False
        print("Recording stopped.")

    def get_latest_volume(self) -> float:
        """
        Returns the volume (RMS) of the most recent audio chunk.
        This can be used for a real-time volume meter.
        """
        with self.lock:
            return self.latest_volume

    def get_wav_data(self) -> bytes:
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
            audio_chunks = list(self.audio_buffer)

        if not audio_chunks:
            return None

        # Perform the potentially slow concatenation outside the lock
        full_audio = np.concatenate(audio_chunks, axis=0)

        # Normalize to 16-bit PCM
        audio_int16 = np.int16(full_audio * 32767)

        # Use an in-memory buffer to write the WAV file
        wav_buffer = io.BytesIO()
        scipy.io.wavfile.write(wav_buffer, self.sample_rate, audio_int16)

        # Go to the beginning of the buffer to read its content
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
        time.sleep(5)
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