import streamlit as st
import os
import tempfile
import time
from main import create_transcript, recognize_speakers
from speaker_manager import save_profile, load_profiles, update_profiles_status
from audio_recorder import list_input_devices, AudioRecorder
from realtime_transcriber import RealtimeTranscriber

# --- Constants ---
PROCESSING_INTERVAL = 3  # seconds

# --- State Management ---

def initialize_session():
    """Initializes the session state ONCE at the start of an app run."""
    if 'app_state' not in st.session_state:
        st.session_state.app_state = 'idle'  # idle, processing_file, recording, stopping
        st.session_state.model_size = 'base'
        st.session_state.transcription_mode = 'File Upload'  # Default mode
        st.session_state.transcript_data = []
        st.session_state.speaker_embeddings = {}
        st.session_state.speaker_name_map = {}
        st.session_state.unrecognized_speakers = {}
        st.session_state.recorder = None
        st.session_state.transcriber = None
        print("Session state fully initialized for the first time.")

def reset_transcript_state():
    """Resets only the data for a new transcription task, preserving user settings like model_size."""
    st.session_state.transcript_data = []
    st.session_state.speaker_embeddings = {}
    st.session_state.speaker_name_map = {}
    st.session_state.unrecognized_speakers = {}
    if 'recorder' in st.session_state and st.session_state.recorder:
        st.session_state.recorder.stop()
    st.session_state.recorder = None
    st.session_state.transcriber = None
    print("Transcript state reset for new task.")

# --- Backend Logic ---

def handle_processing():
    """
    This function acts as a state machine for all backend processing.
    It should be called BEFORE any UI is drawn.
    """
    if st.session_state.app_state == 'processing_file':
        with st.spinner('Processing file...'):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(st.session_state.uploaded_file_name)[1]) as tmp_file:
                    tmp_path = tmp_file.name
                    tmp_file.write(st.session_state.uploaded_file_data)

                transcript, unrecognized, label_map = create_transcript(tmp_path, st.session_state.model_size)

                st.session_state.transcript_data = transcript
                st.session_state.unrecognized_speakers = unrecognized
                st.session_state.speaker_name_map = label_map
                st.success("File processing complete!")

            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)
        st.session_state.app_state = 'idle'

    elif st.session_state.app_state == 'recording':
        try:
            raw_transcript, embeddings = st.session_state.transcriber.process_audio(st.session_state.recorder)
            st.session_state.transcript_data = raw_transcript
            if embeddings:
                st.session_state.speaker_embeddings = embeddings

            active_profiles = {name: data for name, data in load_profiles().items() if data.get('is_active', True)}
            label_map, unrecognized = recognize_speakers(st.session_state.speaker_embeddings, active_profiles)

            st.session_state.speaker_name_map = label_map
            st.session_state.unrecognized_speakers = unrecognized
        except Exception as e:
            st.error(f"An error occurred during real-time transcription: {e}")
            st.session_state.app_state = 'idle'
            if st.session_state.recorder:
                st.session_state.recorder.stop()

    elif st.session_state.app_state == 'stopping':
        if st.session_state.recorder:
            st.session_state.recorder.stop()
            st.session_state.recorder = None
        st.session_state.app_state = 'idle'
        st.info("Recording stopped.")

# --- UI Drawing Functions ---

def format_transcript(transcript_data: list, speaker_name_map: dict) -> str:
    """Formats the transcript data into a readable string."""
    lines = ["--- Transcript ---\n"]
    if not transcript_data:
        return "No transcript generated yet."
    for entry in transcript_data:
        speaker_display = speaker_name_map.get(entry['speaker'], entry['speaker'])
        line = f"[{entry['start']}s - {entry['end']}s] {speaker_display}: {entry['text']}"
        lines.append(line)
    return "\n".join(lines)

def draw_sidebar():
    with st.sidebar:
        st.header("Speaker Profiles")
        st.write("Select speakers to use for recognition.")
        all_profiles = load_profiles()
        if not all_profiles:
            st.info("No saved speaker profiles found.")
        else:
            with st.form(key="speaker_selection_form"):
                speaker_statuses = {name: st.checkbox(name, value=data.get('is_active', True), key=f"speaker_active_{name}") for name, data in sorted(all_profiles.items())}
                if st.form_submit_button("Update Active Speakers"):
                    update_profiles_status(speaker_statuses)
                    st.success("Speaker selection saved.")
                    st.rerun() # Rerun is acceptable here as it's a direct user action with no complex loop.

def draw_main_interface():
    st.header("Transcription Source")

    is_disabled = st.session_state.app_state != 'idle'
    current_model_size_index = ['tiny', 'base', 'small', 'medium', 'large'].index(st.session_state.model_size)

    selected_model = st.selectbox(
        "Select Whisper model size",
        ('tiny', 'base', 'small', 'medium', 'large'),
        index=current_model_size_index,
        key='model_size_selector',
        disabled=is_disabled
    )
    if not is_disabled and selected_model != st.session_state.model_size:
        st.session_state.model_size = selected_model

    # Use a radio button for mode selection to persist the choice across reruns
    mode = st.radio(
        "Choose transcription mode",
        ('File Upload', 'Real-time Recording'),
        key='transcription_mode',
        horizontal=True,
        disabled=is_disabled
    )

    if mode == 'File Upload':
        draw_file_upload_tab(is_disabled)
    else:  # Real-time Recording
        draw_realtime_tab(is_disabled)

def draw_file_upload_tab(is_disabled):
    uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3', 'm4a', 'flac'], disabled=is_disabled)
    if st.button("Create Transcript from File", disabled=is_disabled):
        if uploaded_file:
            reset_transcript_state()
            st.session_state.app_state = 'processing_file'
            st.session_state.uploaded_file_data = uploaded_file.getvalue()
            st.session_state.uploaded_file_name = uploaded_file.name
            # No rerun needed, Streamlit reruns automatically after button press.
        else:
            st.warning("Please upload an audio file.")

def draw_realtime_tab(is_disabled):
    try:
        input_devices = list_input_devices()
        device_options = list(input_devices.keys())
        selected_device_name = st.selectbox("Select an audio input device", options=device_options, key="realtime_device_select", disabled=is_disabled)
    except Exception as e:
        st.error(f"Could not list audio devices. Ensure 'portaudio' is installed. Error: {e}")
        return

    if st.session_state.app_state == 'idle':
        if st.button("Start Recording", key="start_recording"):
            reset_transcript_state()
            device_index = input_devices[selected_device_name]
            st.session_state.transcriber = RealtimeTranscriber(model_size=st.session_state.model_size)
            st.session_state.recorder = AudioRecorder(device_index=device_index)
            st.session_state.recorder.start()
            st.session_state.app_state = 'recording'
            # No rerun needed, Streamlit reruns automatically.

    if st.session_state.app_state == 'recording':
        st.info("🔴 Recording...")
        st.write("Input Volume:")
        volume = st.session_state.recorder.get_latest_volume()
        display_volume = min(int(volume * 500), 100)
        st.progress(display_volume)
        if st.button("Stop Recording", key="stop_recording"):
            st.session_state.app_state = 'stopping'
            # No rerun needed, Streamlit reruns automatically.

def draw_results_and_naming():
    st.header("Transcript Results")

    # Always display the text area, even if empty.
    formatted_transcript_str = format_transcript(st.session_state.transcript_data, st.session_state.speaker_name_map)
    st.text_area("Transcript", formatted_transcript_str, height=300, key="transcript_display_area")

    if st.session_state.transcript_data:
        st.download_button("Download Transcript (.txt)", formatted_transcript_str.encode('utf-8'), "transcript.txt")

    if st.session_state.unrecognized_speakers:
        st.header("Register New Speakers")
        st.write("The following speakers were not found. Assign names and save them.")
        for label in st.session_state.unrecognized_speakers.keys():
            st.text_input(f"Name for speaker '{label}':", key=f"name_for_{label}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reflect Names in Transcript"):
                for label in st.session_state.unrecognized_speakers.keys():
                    new_name = st.session_state.get(f"name_for_{label}")
                    if new_name:
                        st.session_state.speaker_name_map[label] = new_name
                # No rerun needed, Streamlit reruns automatically.
        with col2:
            if st.button("Save Names and Features"):
                saved_count = 0
                speakers_to_remove = []
                for label in list(st.session_state.unrecognized_speakers.keys()):
                    new_name = st.session_state.get(f"name_for_{label}")
                    if new_name:
                        embedding = st.session_state.unrecognized_speakers[label]
                        save_profile(new_name, embedding)
                        st.session_state.speaker_name_map[label] = new_name
                        saved_count += 1
                        speakers_to_remove.append(label)
                if saved_count > 0:
                    for label in speakers_to_remove:
                        del st.session_state.unrecognized_speakers[label]
                    st.success(f"Saved {saved_count} new speaker profiles.")
                    # Rerun to clear the text boxes and update the UI
                    st.rerun()
                else:
                    st.warning("No names were entered to save.")

# --- Main App Execution ---

def main():
    st.title("Transcription Tool with Speaker Diarization")

    initialize_session()

    # This will automatically rerun the page every N seconds if we are recording
    if st.session_state.app_state == 'recording':
        st.components.v1.html(f"<meta http-equiv='refresh' content='{PROCESSING_INTERVAL}'>", height=0)

    # Process state changes from the previous run BEFORE drawing the new UI
    handle_processing()

    draw_sidebar()
    draw_main_interface()
    draw_results_and_naming()

if __name__ == "__main__":
    main()