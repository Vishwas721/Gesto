# Gesture Data Collection Guide

## Overview

The `record_gestures.py` script captures training data for hand gesture recognition. It uses your webcam to record video sequences of hand gestures, normalizes the landmark coordinates, and saves them as `.npy` files.

## Features

- **Real-time hand detection** using MediaPipe Holistic
- **Automatic normalization** of landmarks using Bone-Metric Scaling (scale-invariant)
- **Structured data storage** organized by gesture action and sequence
- **Interactive UI** with visual feedback and countdown timers
- **Robust error handling** and cleanup

## Quick Start

### Prerequisites

Ensure you have the required dependencies:
```bash
pip install opencv-python mediapipe numpy
```

### Running the Script

```bash
python -m backend.data_collection.record_gestures
```

## What the Script Does

1. **Initializes MediaPipe Holistic** for hand, pose, and face landmark detection
2. **Creates directory structure**:
   ```
   MP_Data/
   ├── for_loop/
   │   ├── 0/
   │   │   ├── 0.npy, 1.npy, ..., 29.npy
   │   ├── 1/
   │   └── ...
   ├── function_def/
   └── idle/
   ```

3. **Records 30 sequences per action**:
   - Each sequence is 30 frames of your hand gesture
   - Between sequences, you'll see a 2-second countdown to reset your hand position

4. **Normalizes landmarks on-the-fly**:
   - Translates so the Wrist is at the origin (0, 0, 0)
   - Scales by the Wrist-to-Middle-MCP distance to be scale-invariant
   - Flattens to a (63,) array (21 joints × 3 coordinates)

5. **Saves normalized data**:
   - Each frame is saved as a `.npy` file containing the (63,) normalized array

## Configuration

Edit these values in `record_gestures.py` to customize:

```python
actions = np.array(['for_loop', 'function_def', 'idle'])  # Gesture names
no_sequences = 30           # Videos per action
sequence_length = 30        # Frames per video
SEQUENCE_PAUSE_MS = 2000    # 2-second pause between sequences
```

## Gesture Definitions (MVP)

### For Loop
**Action**: Perform a **circular motion** with your hand
- The circle should be smooth and consistent
- Record 30 sequences of this circular motion

### Function Definition
**Action**: Draw a **box/square shape** in the air
- Make clear corners for the square
- Record 30 sequences of this box drawing

### Idle
**Action**: Let your hand move **randomly** without intent
- This captures the "no gesture" state
- Record 30 sequences of random movement

## UI Guide

During recording, you'll see:

- **Preparation Phase** (2 seconds):
  - Orange text: "Get Ready for {action}"
  - Countdown timer showing seconds remaining

- **Recording Phase** (until 30 frames collected):
  - Green text: "Collecting {action} - Frame X/30"
  - Progress percentage displayed

- **Controls**:
  - Press `Q` to quit at any time

## Output Data Structure

Each saved `.npy` file contains:
- Shape: `(63,)` - representing 21 hand joints × 3 coordinates (x, y, z)
- Dtype: `float32`
- Normalized using Bone-Metric Scaling

### Example Usage

```python
import numpy as np

# Load a single frame
data = np.load("MP_Data/for_loop/0/0.npy")
print(data.shape)  # (63,)

# Load an entire sequence
sequence_data = []
for frame_num in range(30):
    frame = np.load(f"MP_Data/for_loop/0/{frame_num}.npy")
    sequence_data.append(frame)

sequence_array = np.array(sequence_data)  # Shape: (30, 63)
```

## Tips for Best Results

1. **Lighting**: Ensure good lighting for hand detection
2. **Consistency**: Perform gestures at roughly similar speeds
3. **Distance**: Keep your hand in frame throughout the sequence
4. **Background**: Use a neutral background without clutter
5. **Multiple Trials**: Record sequences at different distances and angles for robustness

## Troubleshooting

### "Could not open webcam"
- Check if your webcam is connected and accessible
- Try closing other applications using the camera

### Low hand detection confidence
- Ensure adequate lighting
- Get your entire hand in frame
- Keep hand steady during detection

### Slow performance
- Close other applications
- Reduce camera resolution (edit `cap.set()` calls)
- Reduce `DISPLAY_FPS` if needed

## File Location

Collected data is stored in:
```
backend/data_collection/MP_Data/
```

This structure makes it easy to load data for training your LSTM model.

## Next Steps

Once you've collected data with this script:
1. Use `backend/training/train_gesture_model.py` to train an LSTM
2. Export the trained model for the VS Code extension
3. Connect the backend and frontend together

---

**Note**: The script uses MediaPipe's right-hand tracking. If you want to track the left hand instead, modify the `extract_hand_landmarks()` function to use `results.left_hand_landmarks`.
