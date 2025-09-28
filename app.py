import streamlit as st
import os
import tempfile
import time
from main import create_transcript
from speaker_manager import save_profile, load_profiles, update_profiles_status
from audio_recorder import list_input_devices, AudioRecorder
from realtime_transcriber import RealtimeTranscriber

def format_transcript(transcript: list, speaker_name_map: dict) -> str:
    """
    Formats the transcript data into a readable string, substituting speaker names
    based on the provided map.
    """
    lines = ["--- Meeting Transcript ---\n"]
    for entry in transcript:
        # Use the mapped name; fall back to the original label if not in map
        speaker_display = speaker_name_map.get(entry['speaker'], entry['speaker'])
        line = f"[{entry['start']}s - {entry['end']}s] {speaker_display}: {entry['text']}"
        lines.append(line)
    return "\n".join(lines)

# --- Streamlit UI ---

st.title("Transcription Tool with Speaker Diarization")
st.write("Upload an audio file to perform transcription with speaker recognition.")

# --- Speaker Selection Sidebar ---
with st.sidebar:
    st.header("Speaker Selection")
    st.write("Select the speakers to use for transcription.")
    
    all_profiles = load_profiles()
    
    if not all_profiles:
        st.info("No saved speaker profiles found.")
    else:
        with st.form(key="speaker_selection_form"):
            speaker_statuses = {}
            for name, data in sorted(all_profiles.items()):
                is_active = st.checkbox(name, value=data.get('is_active', True), key=f"speaker_active_{name}")
                speaker_statuses[name] = is_active
            
            if st.form_submit_button("Save Selection"):
                try:
                    update_profiles_status(speaker_statuses)
                    st.success("Speaker selection saved successfully.")
                except Exception as e:
                    st.error(f"An error occurred while saving: {e}")

# --- Session State Initialization ---
if 'transcript_data' not in st.session_state:
    st.session_state.transcript_data = None
if 'unrecognized_speakers' not in st.session_state:
    st.session_state.unrecognized_speakers = None
if 'speaker_name_map' not in st.session_state:
    st.session_state.speaker_name_map = {}

# --- UI Components ---

uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3', 'm4a', 'flac'])
model_size = st.selectbox("Select Whisper model size", ('tiny', 'base', 'small', 'medium', 'large'), index=1)

if st.button("Create Transcript"):
    if uploaded_file is not None:
        # Reset state for a new run
        st.session_state.transcript_data = None
        st.session_state.unrecognized_speakers = None
        st.session_state.speaker_name_map = {}

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(uploaded_file.getvalue())

        st.info("Processing started...")
        with st.spinner('Performing diarization, speaker recognition, and transcription...'):
            try:
                # Backend now returns three items
                transcript_data, unrecognized, label_map = create_transcript(tmp_path, model_size)
                
                if transcript_data is not None:
                    st.session_state.transcript_data = transcript_data
                    st.session_state.unrecognized_speakers = unrecognized
                    st.session_state.speaker_name_map = label_map # Contains recognized names and original labels
                    st.success("Processing complete!")
                else:
                    st.error("Failed to create transcript.")
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    else:
        st.warning("Please upload an audio file.")

# --- Display Results ---
if st.session_state.transcript_data:
    st.header("Transcript Results")

    # Format and display the transcript using the current name map
    formatted_transcript = format_transcript(
        st.session_state.transcript_data,
        st.session_state.speaker_name_map
    )
    st.text_area("Transcript", formatted_transcript, height=300)
    st.download_button("Download Transcript (.txt)", formatted_transcript.encode('utf-8'), "transcript.txt")

# --- Speaker Naming and Saving Section ---
if st.session_state.unrecognized_speakers and len(st.session_state.unrecognized_speakers) > 0:
    st.header("Register New Speakers")
    st.write("The following speakers were not found in profiles. Please enter their names, reflect the changes, and then save.")

    # Create text inputs for each unrecognized speaker
    # The user's input is stored in st.session_state with the specified key.
    for label in st.session_state.unrecognized_speakers.keys():
        st.text_input(
            f"Name for speaker '{label}':",
            key=f"name_for_{label}"
        )

    # Create two columns for the buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Reflect Names in Transcript"):
            # Read the current values from the text inputs and update the name map
            for label in st.session_state.unrecognized_speakers.keys():
                new_name = st.session_state[f"name_for_{label}"]
                if new_name:
                    st.session_state.speaker_name_map[label] = new_name
            st.success("Transcript display updated.")
            st.rerun()

    with col2:
        if st.button("Save Names and Features to JSON"):
            saved_count = 0
            speakers_to_remove = []
            # Read the current values from the text inputs to save them
            for label in list(st.session_state.unrecognized_speakers.keys()):
                new_name = st.session_state[f"name_for_{label}"]
                if new_name:
                    embedding = st.session_state.unrecognized_speakers[label]
                    save_profile(new_name, embedding)
                    
                    st.session_state.speaker_name_map[label] = new_name
                    saved_count += 1
                    speakers_to_remove.append(label)

            if saved_count > 0:
                # Remove the saved speakers from the "unrecognized" list
                for label in speakers_to_remove:
                    del st.session_state.unrecognized_speakers[label]
                
                st.success(f"Saved {saved_count} speaker profiles.")
                st.rerun()
            else:
                st.warning("No names were entered to save.")


# --- Real-time Transcription Section ---
st.header("Real-time Transcription")
st.write(
    "Select an audio device and start recording. The transcript will be generated in real-time. "
    "Note: System audio on macOS may require a virtual audio device like BlackHole."
)

# Initialize session state for real-time components
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'recorder' not in st.session_state:
    st.session_state.recorder = None
if 'transcriber' not in st.session_state:
    st.session_state.transcriber = None
if 'realtime_transcript_display' not in st.session_state:
    st.session_state.realtime_transcript_display = ""
# This flag controls the periodic rerun loop
if 'run_realtime_loop' not in st.session_state:
    st.session_state.run_realtime_loop = False

# UI for Real-time Section
try:
    input_devices = list_input_devices()
except Exception as e:
    input_devices = None
    st.error(f"Could not list audio devices. Please ensure 'portaudio' is installed (e.g., `sudo apt-get install portaudio19-dev` or `brew install portaudio`). Error: {e}")

if input_devices:
    device_options = list(input_devices.keys())
    selected_device_name = st.selectbox(
        "Select an audio input device", options=device_options, key="realtime_device_select"
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start Recording", key="start_recording_btn", disabled=st.session_state.is_recording):
            device_index = input_devices[selected_device_name]
            # Use the model size selected in the file-based UI section
            st.session_state.transcriber = RealtimeTranscriber(model_size=model_size)
            st.session_state.recorder = AudioRecorder(device_index=device_index)

            st.session_state.recorder.start()
            st.session_state.is_recording = True
            st.session_state.run_realtime_loop = True
            st.rerun()

    with col2:
        if st.button("Stop Recording", key="stop_recording_btn", disabled=not st.session_state.is_recording):
            st.session_state.run_realtime_loop = False  # Signal the loop to stop
            if st.session_state.recorder:
                st.session_state.recorder.stop()
            st.session_state.is_recording = False
            st.info("Recording stopped. Finalizing transcript...")

            # Perform one last processing call to get any remaining audio
            if st.session_state.transcriber and st.session_state.recorder:
                final_transcript_data = st.session_state.transcriber.process_audio(st.session_state.recorder)
                lines = [f"[{e['start']}s - {e['end']}s] {e['speaker']}: {e['text']}" for e in final_transcript_data]
                st.session_state.realtime_transcript_display = "\n".join(lines)

            st.rerun()

    # Display area for real-time transcript
    st.write("Live Transcript:")
    transcript_placeholder = st.empty()
    transcript_placeholder.text_area(
        "Transcript will appear here...",
        value=st.session_state.realtime_transcript_display,
        height=300,
        key="realtime_transcript_area"
    )

    # The "pseudo-real-time" loop logic, controlled by the run_realtime_loop flag
    if st.session_state.get('run_realtime_loop', False):
        st.info("🔴 Recording... Transcript will update every 5 seconds.")

        try:
            if st.session_state.transcriber and st.session_state.recorder:
                current_transcript_data = st.session_state.transcriber.process_audio(st.session_state.recorder)
                lines = [f"[{e['start']}s - {e['end']}s] {e['speaker']}: {e['text']}" for e in current_transcript_data]
                st.session_state.realtime_transcript_display = "\n".join(lines)

            # Wait and then trigger a rerun to refresh the UI and run this block again
            time.sleep(5)
            st.rerun()

        except Exception as e:
            st.error(f"An error occurred during real-time transcription: {e}")
            st.session_state.run_realtime_loop = False
            st.session_state.is_recording = False
            if st.session_state.recorder:
                st.session_state.recorder.stop()
            st.rerun()
