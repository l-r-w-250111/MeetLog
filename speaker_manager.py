import json
import os
import numpy as np

PROFILES_FILE = "speaker_profiles.json"

def load_profiles() -> dict:
    """
    Loads speaker profiles from the JSON file with UTF-8 encoding.
    Returns a dictionary of profiles, or an empty dict if the file doesn't exist.
    """
    if os.path.exists(PROFILES_FILE):
        try:
            with open(PROFILES_FILE, 'r', encoding='utf-8') as f:
                profiles = json.load(f)
                # Convert embedding lists back to numpy arrays and ensure 'is_active' exists
                for name, data in profiles.items():
                    if 'embedding' in data and isinstance(data['embedding'], list):
                        data['embedding'] = np.array(data['embedding'])
                    # For backward compatibility, default is_active to True if not present
                    if 'is_active' not in data:
                        data['is_active'] = True
                return profiles
        except (json.JSONDecodeError, UnicodeDecodeError):
            # If file is corrupted or not valid UTF-8, handle it gracefully
            print(f"Warning: Could not decode {PROFILES_FILE}. It might be corrupted. Starting with empty profiles.")
            return {}
    return {}

def _save_profiles_to_file(profiles: dict):
    """
    Internal helper to save the complete profiles dictionary to a file,
    handling serialization and ensuring UTF-8 encoding.
    """
    profiles_to_save = {}
    for speaker_name, data in profiles.items():
        # Create a copy to avoid modifying the original dict in memory
        data_to_save = data.copy()
        # Ensure embedding is a list before saving
        if 'embedding' in data_to_save and isinstance(data_to_save['embedding'], np.ndarray):
            data_to_save['embedding'] = data_to_save['embedding'].tolist()
        profiles_to_save[speaker_name] = data_to_save
            
    with open(PROFILES_FILE, 'w', encoding='utf-8') as f:
        json.dump(profiles_to_save, f, indent=4, ensure_ascii=False)

def save_profile(name: str, embedding: np.ndarray):
    """
    Saves a new speaker profile or updates an existing one using UTF-8 encoding.
    New profiles are set to active by default.
    """
    profiles = load_profiles()
    
    # Add or update the profile for the new speaker, defaulting is_active to True
    profiles[name] = {"embedding": embedding, "is_active": True}
    
    _save_profiles_to_file(profiles)

def update_profiles_status(statuses: dict):
    """
    Updates the 'is_active' status for multiple speakers.

    Args:
        statuses: A dictionary where keys are speaker names and values are booleans.
    """
    profiles = load_profiles()
    
    for name, is_active in statuses.items():
        if name in profiles:
            profiles[name]['is_active'] = is_active
            
    _save_profiles_to_file(profiles)

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Calculates the cosine similarity between two embedding vectors.
    """
    emb1 = emb1.astype(float)
    emb2 = emb2.astype(float)
    
    dot_product = np.dot(emb1, emb2)
    norm_emb1 = np.linalg.norm(emb1)
    norm_emb2 = np.linalg.norm(emb2)
    
    if norm_emb1 == 0 or norm_emb2 == 0:
        return 0.0
        
    return dot_product / (norm_emb1 * norm_emb2)

def find_matching_speaker(new_embedding: np.ndarray, profiles: dict, threshold: float = 0.75) -> str:
    """
    Finds a matching speaker from saved profiles based on cosine similarity.
    """
    best_match_name = None
    max_similarity = -1.0
    
    for name, data in profiles.items():
        saved_embedding = data.get('embedding')
        if saved_embedding is not None:
            if isinstance(saved_embedding, list):
                saved_embedding = np.array(saved_embedding)
            
            similarity = cosine_similarity(new_embedding, saved_embedding)
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_name = name
    
    if max_similarity >= threshold:
        print(f"Found match for new speaker: {best_match_name} with similarity {max_similarity:.2f}")
        return best_match_name
    
    return None