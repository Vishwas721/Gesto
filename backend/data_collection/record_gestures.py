"""
Gesture Data Collection Script

This script captures video sequences of hand gestures, normalizes the landmarks,
and saves them to disk for training the gesture recognition LSTM model.

Usage:
    python -m backend.data_collection.record_gestures

The script will:
1. Open a webcam feed
2. Loop through each gesture action
3. For each action, record 30 video sequences
4. For each sequence, capture 30 frames
5. Normalize landmarks using Bone-Metric Scaling
6. Save normalized data as .npy files

Directory Structure Created:
    MP_Data/
    ├── for_loop/
    │   ├── 0/
    │   │   ├── 0.npy
    │   │   ├── 1.npy
    │   │   └── ...
    │   ├── 1/
    │   └── ...
    ├── function_def/
    └── idle/
"""

import cv2
import numpy as np
import os
import mediapipe as mp
from pathlib import Path
import sys

# Import the normalization function
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.utils.normalization import normalize_landmarks


# ============================================================================
# CONFIGURATION
# ============================================================================

# Define the root path for storing collected data
MP_DATA_PATH = Path(__file__).parent / "MP_Data"

# List of gesture actions to record
actions = np.array(['for_loop', 'function_def', 'idle'])

# Number of videos (sequences) to record per action
no_sequences = 30

# Number of frames to capture per video
sequence_length = 30

# Pause duration (milliseconds) between sequences to allow hand reset
SEQUENCE_PAUSE_MS = 2000

# Window display settings
WINDOW_NAME = "Gesto - Gesture Data Collection"
DISPLAY_FPS = 30


# ============================================================================
# MEDIAPIPE INITIALIZATION
# ============================================================================

def initialize_mediapipe():
    """
    Initialize MediaPipe Holistic model for hand, pose, and face detection.
    
    Returns:
        holistic: MediaPipe Holistic instance
        mp_drawing: MediaPipe drawing utilities
    """
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    return holistic, mp_drawing


# ============================================================================
# DIRECTORY SETUP
# ============================================================================

def setup_directories():
    """
    Create the MP_Data directory structure for storing collected data.
    """
    # Create root MP_Data directory
    MP_DATA_PATH.mkdir(parents=True, exist_ok=True)
    print(f"✓ Root directory created: {MP_DATA_PATH}")
    
    # Create subdirectories for each action
    for action in actions:
        action_path = MP_DATA_PATH / action
        action_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Action directory created: {action_path}")
        
        # Create subdirectories for each sequence
        for seq_num in range(no_sequences):
            seq_path = action_path / str(seq_num)
            seq_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n✓ Directory structure setup complete!")
    print(f"  Total: {len(actions)} actions × {no_sequences} sequences × {sequence_length} frames\n")


# ============================================================================
# LANDMARK EXTRACTION AND PROCESSING
# ============================================================================

def extract_hand_landmarks(results):
    """
    Extract hand landmarks from MediaPipe detection results.
    
    Args:
        results: MediaPipe Holistic detection results
    
    Returns:
        landmarks: (21, 3) array of hand landmarks, or None if not detected
    """
    if results.right_hand_landmarks is None:
        return None
    
    # Extract right hand landmarks
    landmarks = np.array([
        [lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark
    ], dtype=np.float32)
    
    return landmarks


def process_landmarks(landmarks):
    """
    Process raw landmarks through normalization.
    
    Args:
        landmarks: (21, 3) array of hand landmarks, or None
    
    Returns:
        normalized: (63,) array of normalized landmarks, or zero array if not detected
    """
    if landmarks is None:
        # Return zero array if no hand detected
        return np.zeros((63,), dtype=np.float32)
    
    try:
        # Apply normalization
        normalized = normalize_landmarks(landmarks)
        return normalized.astype(np.float32)
    except Exception as e:
        print(f"  Warning: Normalization failed: {e}")
        return np.zeros((63,), dtype=np.float32)


# ============================================================================
# VIDEO CAPTURE AND DISPLAY
# ============================================================================

def draw_status_text(frame, text, color=(0, 255, 0), position="top"):
    """
    Draw status text on the video frame.
    
    Args:
        frame: Input image frame
        text: Text to display
        color: BGR color tuple
        position: 'top' or 'bottom'
    
    Returns:
        frame: Frame with text overlay
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    
    # Get text size to center it
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    frame_h, frame_w = frame.shape[:2]
    
    if position == "top":
        x = (frame_w - text_size[0]) // 2
        y = 50
    else:  # bottom
        x = (frame_w - text_size[0]) // 2
        y = frame_h - 30
    
    # Add background rectangle for better visibility
    bg_padding = 10
    cv2.rectangle(frame, 
                  (x - bg_padding, y - text_size[1] - bg_padding),
                  (x + text_size[0] + bg_padding, y + bg_padding),
                  (0, 0, 0), -1)
    
    # Put text
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)
    
    return frame


# ============================================================================
# MAIN DATA COLLECTION LOOP
# ============================================================================

def collect_gesture_data():
    """
    Main function to collect gesture training data.
    """
    print("\n" + "=" * 70)
    print("GESTO - GESTURE DATA COLLECTION")
    print("=" * 70)
    
    # Setup directories
    setup_directories()
    
    # Initialize MediaPipe
    print("Initializing MediaPipe Holistic model...")
    holistic, mp_drawing = initialize_mediapipe()
    print("✓ MediaPipe initialized\n")
    
    # Initialize webcam
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    # Set camera resolution for better performance
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
            print(f"{'=' * 70}")
            
            for sequence_idx in range(no_sequences):
                print(f"  Sequence {sequence_idx + 1}/{no_sequences}...", end=" ", flush=True)
                
                # Wait for 2 seconds before starting sequence collection
                # This allows the user to prepare their hand gesture
                print("(preparing)", end="", flush=True)
                
                frame_count = 0
                start_wait_time = cv2.getTickCount()
                
                while frame_count < SEQUENCE_PAUSE_MS:
                    success, frame = cap.read()
                    if not success:
                        break
                    
                    # Flip frame for selfie view
                    frame = cv2.flip(frame, 1)
                    
                    # Display preparation message
                    frame = draw_status_text(
                        frame,
                        f"Get Ready for {action} (seq {sequence_idx + 1})",
                        color=(0, 165, 255),  # Orange
                        position="top"
                    )
                    
                    # Show countdown
                    elapsed = int((cv2.getTickCount() - start_wait_time) / cv2.getTickFrequency() * 1000)
                    remaining = max(0, SEQUENCE_PAUSE_MS - elapsed)
                    frame = draw_status_text(
                        frame,
                        f"Starting in {remaining // 1000 + 1}s",
                        color=(0, 165, 255),
                        position="bottom"
                    )
                    
                    cv2.imshow(WINDOW_NAME, frame)
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord('q'):
                        raise KeyboardInterrupt("User quit")
                    
                    frame_count = int((cv2.getTickCount() - start_wait_time) / cv2.getTickFrequency() * 1000)
                
                # Now collect frames for this sequence
                frames_collected = 0
                frame_start_time = cv2.getTickCount()
                
                while frames_collected < sequence_length:
                    success, frame = cap.read()
                    if not success:
                        break
                    
                    # Flip frame for selfie view
                    frame = cv2.flip(frame, 1)
                    
                    # Process frame with MediaPipe
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(frame_rgb)
                    
                    # Extract and normalize landmarks
                    landmarks = extract_hand_landmarks(results)
                    normalized_data = process_landmarks(landmarks)
                    
                    # Save normalized data
                    save_path = MP_DATA_PATH / action / str(sequence_idx) / f"{frames_collected}.npy"
                    np.save(str(save_path), normalized_data)
                    
                    # Display collection message
                    frame = draw_status_text(
                        frame,
                        f"Collecting {action} - Frame {frames_collected + 1}/{sequence_length}",
                        color=(0, 255, 0),  # Green
                        position="top"
                    )
                    
                    # Show progress bar
                    progress = frames_collected / sequence_length
                    frame = draw_status_text(
                        frame,
                        f"Progress: {int(progress * 100)}%",
                        color=(0, 255, 0),
                        position="bottom"
                    )
                    
                    cv2.imshow(WINDOW_NAME, frame)
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord('q'):
                        raise KeyboardInterrupt("User quit")
                    
                    frames_collected += 1
                
                print("✓")
        
        print(f"\n{'=' * 70}")
        print("✓ DATA COLLECTION COMPLETE!")
        print(f"{'=' * 70}")
        print(f"\nSaved to: {MP_DATA_PATH}")
        print(f"Total files created: {len(actions)} × {no_sequences} × {sequence_length} = {len(actions) * no_sequences * sequence_length} .npy files")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Data collection interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during data collection: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\nCleaning up...")
        holistic.close()
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Cleanup complete")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    collect_gesture_data()
