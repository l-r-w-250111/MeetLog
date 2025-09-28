import streamlit as st
import os
import tempfile
from main import create_transcript
from speaker_manager import save_profile, load_profiles, update_profiles_status

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

