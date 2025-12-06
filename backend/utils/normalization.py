import numpy as np
from typing import Union, List


def normalize_landmarks(landmarks: Union[List, np.ndarray]) -> np.ndarray:
    """
    Normalizes MediaPipe hand landmarks to be scale-invariant using Bone-Metric Scaling.
    
    The normalization process:
    1. Translates so that the Wrist (Index 0) is at the origin (0, 0, 0)
    2. Scales by dividing all coordinates by the distance between Wrist (Index 0) 
       and Middle Finger MCP (Index 9)
    
    Args:
        landmarks: Either a list of MediaPipe landmark objects or a (21, 3) NumPy array
                   representing hand landmarks in 3D space
    
    Returns:
        A flattened NumPy array of shape (63,) containing normalized coordinates
        (21 points × 3 coordinates)
    
    Raises:
        ValueError: If landmarks do not contain at least 21 points
    """
    # Convert to NumPy array if necessary
    if not isinstance(landmarks, np.ndarray):
        # Handle MediaPipe landmark objects
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    else:
        coords = np.array(landmarks, dtype=np.float32)
    
    # Validate input shape
    if coords.shape[0] != 21:
        raise ValueError(f"Expected 21 landmarks, got {coords.shape[0]}")
    if coords.shape[1] != 3:
        raise ValueError(f"Expected 3 coordinates per landmark, got {coords.shape[1]}")
    
    # Step A: Translation - Move Wrist (Index 0) to origin
    wrist = coords[0]
    translated = coords - wrist
    
    # Step B: Calculate reference bone distance (Wrist to Middle MCP)
    # Wrist: Index 0, Middle MCP: Index 9
    reference_vector = translated[9] - translated[0]  # Should be [0,0,0] to [x,y,z] after translation
    reference_distance = np.linalg.norm(reference_vector)
    
    # Step C: Scale by dividing by reference distance
    if reference_distance == 0:
        raise ValueError("Reference bone distance is zero. Hand landmarks may be invalid.")
    
    normalized = translated / reference_distance
    
    # Flatten to shape (63,)
    flattened = normalized.flatten()
    
    return flattened
