"""
Gesture Data Collection Script - Hand Detection Edition

This script captures video sequences of hand gestures using MediaPipe Hands,
normalizes the landmarks, and saves them to disk for training.

Uses mp.solutions.hands.Hands for more stable hand detection.

Usage:
    python -m backend.data_collection.record_gestures
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import shutil
import sys
from time import time

# Import normalization function
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.utils.normalization import normalize_landmarks


# ============================================================================
# CONFIGURATION
# ============================================================================

# Data storage path
MP_DATA_PATH = Path(__file__).parent / "MP_Data"

# Gesture actions to record
actions = np.array(['for_loop', 'idle'])

# Sequences per action
no_sequences = 100

# Frames per sequence
sequence_length = 30

# Pause between sequences (ms) for hand reset
SEQUENCE_PAUSE_MS = 2000

# MediaPipe Hands configuration
HANDS_CONFIG = {
    'static_image_mode': False,
    'max_num_hands': 1,
    'min_detection_confidence': 0.7,
    'min_tracking_confidence': 0.5
}

# Display settings
WINDOW_NAME = "Gesto - Hand Gesture Recorder"


# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_mediapipe():
    """Initialize MediaPipe Hands and drawing utilities."""
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    hands = mp_hands.Hands(
        static_image_mode=HANDS_CONFIG['static_image_mode'],
        max_num_hands=HANDS_CONFIG['max_num_hands'],
        min_detection_confidence=HANDS_CONFIG['min_detection_confidence'],
        min_tracking_confidence=HANDS_CONFIG['min_tracking_confidence']
    )
    
    return hands, mp_drawing


# ============================================================================
# DIRECTORY SETUP
# ============================================================================

def setup_directories():
    """Create MP_Data directory structure, overwriting old data."""
    # Remove old data if exists
    if MP_DATA_PATH.exists():
        shutil.rmtree(MP_DATA_PATH)
        print(f"✓ Removed old data directory")
    
    # Create root directory
    MP_DATA_PATH.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created root directory: {MP_DATA_PATH}")
    
    # Create action subdirectories and sequence folders
    for action in actions:
        action_path = MP_DATA_PATH / action
        action_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Action '{action}'")
        
        for seq_num in range(no_sequences):
            seq_path = action_path / str(seq_num)
            seq_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n✓ Directory structure ready")
    print(f"  Will record {no_sequences} sequences × {len(actions)} actions")
    print(f"  Each sequence: {sequence_length} frames\n")


# ============================================================================
# HAND DETECTION & NORMALIZATION
# ============================================================================

def extract_hand_keypoints(results):
    """
    Extract hand keypoints from MediaPipe results.
    
    Args:
        results: MediaPipe Hands detection results
    
    Returns:
        Normalized (63,) keypoint array, or None if no hand detected
    """
    # Check if hand was detected
    if not results.multi_hand_landmarks:
        return None
    
    # Extract first hand detected
    hand_landmarks = results.multi_hand_landmarks[0]
    
    # Convert to numpy array (21 landmarks × 3 coords = 63 values)
    keypoints = np.array([
        [landmark.x, landmark.y, landmark.z]
        for landmark in hand_landmarks.landmark
    ]).flatten()
    
    # Normalize using bone-metric scaling
    keypoints = normalize_landmarks(keypoints)
    
    return keypoints


def is_valid_data(keypoints):
    """
    Check if keypoint data is valid (not all zeros).
    
    Args:
        keypoints: (63,) normalized keypoint array
    
    Returns:
        True if valid, False if all zeros (hand not detected)
    """
    if keypoints is None:
        return False
    
    return np.max(np.abs(keypoints)) > 0


# ============================================================================
# DRAWING UTILITIES
# ============================================================================

def draw_hand_skeleton(frame, results, mp_drawing):
    """Draw hand landmarks and connections on frame."""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )


def draw_status(frame, text, position=(10, 30), color=(0, 255, 0), size=0.7):
    """Draw status text on frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, position, font, size, color, 2)
    return frame


def draw_error(frame, text, position=(10, 60)):
    """Draw error text in red."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, position, font, 1.0, (0, 0, 255), 3)
    return frame


# ============================================================================
# MAIN COLLECTION LOOP
# ============================================================================

def collect_gesture_data():
    """Main function to collect gesture data."""
    
    # Safety confirmation
    print("\n" + "=" * 70)
    print("⚠️  WARNING: This will delete all old data in MP_Data/")
    print("=" * 70)
    input("Press Enter to continue or Ctrl+C to cancel...\n")
    
    print("=" * 70)
    print("GESTO - HAND GESTURE DATA COLLECTION")
    print("=" * 70)
    
    # Setup directories
    setup_directories()
    
    # Initialize MediaPipe
    print("Initializing MediaPipe Hands...")
    hands, mp_drawing = initialize_mediapipe()
    print("✓ MediaPipe initialized\n")
    
    # Initialize webcam
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✓ Webcam initialized\n")
    
    # Create display window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)
    
    try:
        # Main collection loop
        for action_idx, action in enumerate(actions):
            print(f"\n{'=' * 70}")
            print(f"ACTION {action_idx + 1}/{len(actions)}: {action.upper()}")
            print(f"Recording sequences 0-{no_sequences - 1}")
            print(f"{'=' * 70}\n")
            
            for sequence_idx in range(no_sequences):
                print(f"  Sequence {sequence_idx}/{no_sequences}...", end=" ", flush=True)
                
                # Wait for hand reset (preparation phase)
                print("(preparing)", end="", flush=True)
                
                start_wait_time = time()
                while (time() - start_wait_time) * 1000 < SEQUENCE_PAUSE_MS:
                    success, frame = cap.read()
                    if not success:
                        break
                    
                    frame = cv2.flip(frame, 1)
                    
                    # Detect hand
                    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw_hand_skeleton(frame, results, mp_drawing)
                    
                    # Show preparation message
                    frame = draw_status(frame, "PREPARE YOUR HAND", (10, 30), (0, 255, 0))
                    elapsed_ms = int((time() - start_wait_time) * 1000)
                    remaining_s = max(0, 2 - (elapsed_ms // 1000))
                    frame = draw_status(frame, f"Starting in {remaining_s}s...", (10, 70), (0, 255, 0))
                    
                    if not results.multi_hand_landmarks:
                        frame = draw_error(frame, "NO HAND DETECTED", (350, 360))
                    
                    cv2.imshow(WINDOW_NAME, frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt
                
                # Recording phase
                sequence_data = []
                frames_recorded = 0
                
                while frames_recorded < sequence_length:
                    success, frame = cap.read()
                    if not success:
                        break
                    
                    frame = cv2.flip(frame, 1)
                    
                    # Detect hand
                    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw_hand_skeleton(frame, results, mp_drawing)
                    
                    # Extract and validate keypoints
                    keypoints = extract_hand_keypoints(results)
                    
                    if is_valid_data(keypoints):
                        # Valid frame - save it
                        sequence_data.append(keypoints)
                        frames_recorded += 1
                        
                        # Show recording status
                        frame = draw_status(
                            frame,
                            f"Recording: Sequence {sequence_idx}/100, Frame {frames_recorded}/{sequence_length}",
                            (10, 30),
                            (0, 255, 0)
                        )
                    else:
                        # Invalid frame - skip it
                        frame = draw_error(frame, "SKIPPING FRAME - NO HAND", (250, 360))
                        
                        # Show status anyway
                        frame = draw_status(
                            frame,
                            f"Waiting for hand... ({frames_recorded}/{sequence_length} frames captured)",
                            (10, 30),
                            (0, 165, 255)
                        )
                    
                    cv2.imshow(WINDOW_NAME, frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt
                
                # Save sequence to disk
                if len(sequence_data) == sequence_length:
                    action_path = MP_DATA_PATH / action
                    seq_path = action_path / str(sequence_idx)
                    
                    for frame_idx, keypoints in enumerate(sequence_data):
                        frame_path = seq_path / f"{frame_idx}.npy"
                        np.save(frame_path, keypoints)
                    
                    print(" ✓ SAVED")
                else:
                    print(f" ❌ FAILED (only {len(sequence_data)}/{sequence_length} frames)")
        
        # Summary
        print(f"\n{'=' * 70}")
        print("✓ DATA COLLECTION COMPLETE!")
        print(f"{'=' * 70}")
        print(f"\nSaved to: {MP_DATA_PATH}")
        print(f"Total sequences: {len(actions)} × {no_sequences} = {len(actions) * no_sequences}")
        print(f"Total files: {len(actions) * no_sequences * sequence_length} .npy files")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Data collection interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during collection: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nCleaning up...")
        hands.close()
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Cleanup complete")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    collect_gesture_data()
