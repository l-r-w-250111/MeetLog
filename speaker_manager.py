import json
import os
import numpy as np

PROFILES_FILE = "speaker_profiles.json"

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for numpy types.
    Converts numpy integers, floats, and arrays to their Python equivalents.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

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
                        data['embedding'] = np.array(data['embedding'], dtype=np.float32)
                    if 'is_active' not in data:
                        data['is_active'] = True
                return profiles
        except (json.JSONDecodeError, UnicodeDecodeError):
            print(f"Warning: Could not decode {PROFILES_FILE}. It might be corrupted. Starting with empty profiles.")
            return {}
    return {}

def _save_profiles_to_file(profiles: dict):
    """
    Internal helper to save the complete profiles dictionary to a file,
    using a custom NumpyEncoder to handle numpy data types.
    """
    with open(PROFILES_FILE, 'w', encoding='utf-8') as f:
        json.dump(profiles, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

def save_profile(name: str, embedding: np.ndarray):
    """
    Saves a new speaker profile or updates an existing one using UTF-8 encoding.
    New profiles are set to active by default.
    """
    profiles = load_profiles()
    
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
    Ensures both embeddings are float32 to prevent dtype mismatch issues.
    """
    # [FIX] Force conversion to float32 to avoid potential dtype mismatches
    # (e.g., comparing float64 from live processing with float32 from loaded file)
    emb1 = emb1.astype(np.float32)
    emb2 = emb2.astype(np.float32)
    
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
                # This case should ideally not be hit if load_profiles is always used
                saved_embedding = np.array(saved_embedding, dtype=np.float32)
            
            similarity = cosine_similarity(new_embedding, saved_embedding)
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_name = name
    
    if max_similarity >= threshold:
        print(f"Found match for new speaker: {best_match_name} with similarity {max_similarity:.2f}")
        return best_match_name
    
    return None
