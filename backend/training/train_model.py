"""Train an LSTM gesture classifier on normalized hand landmark data.

Usage:
    python -m backend.training.train_model

Assumes data are stored under backend/data_collection/MP_Data in the layout:
    MP_Data/{action}/{sequence}/{frame}.npy
where each frame is a (63,) normalized landmark vector.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard


# Configuration
ACTIONS = np.array(["for_loop", "idle"], dtype=str)
LABEL_MAP: Dict[str, int] = {action: idx for idx, action in enumerate(ACTIONS)}
SEQUENCE_LENGTH = 30
TEST_SIZE = 0.05
EPOCHS = 500
DATA_PATH = Path(__file__).resolve().parent.parent / "data_collection" / "MP_Data"
MODEL_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "action.h5"
LOGS_PATH = Path(__file__).resolve().parent.parent / "logs"


def load_sequences(
    data_path: Path = DATA_PATH,
    actions: np.ndarray = ACTIONS,
    sequence_length: int = SEQUENCE_LENGTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load sequences and labels from disk."""
    sequences: List[List[np.ndarray]] = []
    labels: List[int] = []

    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    for action in actions:
        action_dir = data_path / action
        if not action_dir.exists():
            raise FileNotFoundError(f"Missing action directory: {action_dir}")

        sequence_dirs = sorted(
            [p for p in action_dir.iterdir() if p.is_dir()],
            key=lambda p: int(p.name) if p.name.isdigit() else p.name,
        )

        for seq_dir in sequence_dirs:
            frames: List[np.ndarray] = []
            for frame_idx in range(sequence_length):
                frame_path = seq_dir / f"{frame_idx}.npy"
                if not frame_path.exists():
                    frames = []  # mark incomplete sequence
                    break
                frames.append(np.load(frame_path))

            if len(frames) == sequence_length:
                sequences.append(frames)
                labels.append(LABEL_MAP[action])

    if not sequences:
        raise RuntimeError("No complete sequences found. Check your data layout.")

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y


def build_model(num_actions: int) -> Sequential:
    """Build the simplified LSTM classification model."""
    model = Sequential(
        [
            LSTM(
                64,
                return_sequences=True,
                activation="relu",
                input_shape=(SEQUENCE_LENGTH, 63),
            ),
            LSTM(32, return_sequences=False, activation="relu"),
            Dense(32, activation="relu"),
            Dense(num_actions, activation="softmax"),
        ]
    )
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    return model


def train():
    print("Loading data from:", DATA_PATH)
    X, y = load_sequences()
    print(f"Loaded {X.shape[0]} sequences. Shape: {X.shape}")

    # Train/validation split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, shuffle=True
    )
    y_train_cat = to_categorical(y_train).astype(np.float32)
    y_test_cat = to_categorical(y_test).astype(np.float32)

    print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_test.shape[0]}")

    # Build and train model
    model = build_model(num_actions=len(ACTIONS))
    
    # Create TensorBoard callback for visualization
    LOGS_PATH.mkdir(parents=True, exist_ok=True)
    tensorboard_callback = TensorBoard(
        log_dir=str(LOGS_PATH),
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    
    print(f"\nTraining for {EPOCHS} epochs (simplified architecture)...")
    print("Success criteria: categorical_accuracy >= 0.80 and loss < 0.5")
    print()
    
    history = model.fit(
        X_train,
        y_train_cat,
        epochs=EPOCHS,
        validation_data=(X_test, y_test_cat),
        callbacks=[tensorboard_callback],
        verbose=1,
    )

    # Save model
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_OUTPUT_PATH))
    print(f"\nModel saved to: {MODEL_OUTPUT_PATH}")
    
    # Print final metrics
    final_loss = history.history['loss'][-1]
    final_acc = history.history['categorical_accuracy'][-1]
    val_loss = history.history['val_loss'][-1]
    val_acc = history.history['val_categorical_accuracy'][-1]
    
    print(f"\nFinal Training Metrics:")
    print(f"  Loss: {final_loss:.4f}")
    print(f"  Categorical Accuracy: {final_acc:.4f}")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Val Categorical Accuracy: {val_acc:.4f}")
    
    if final_acc >= 0.80 and final_loss < 0.5:
        print("\n[SUCCESS] SUCCESS CRITERIA MET!")
        print("  The model has learned well. Ready for inference.")
    else:
        print("\n[FAILED] SUCCESS CRITERIA NOT MET")
        if final_acc < 0.80:
            print(f"  - Categorical accuracy {final_acc:.4f} is below 0.80 threshold")
        if final_loss >= 0.5:
            print(f"  - Loss {final_loss:.4f} is above 0.5 threshold")
        print("\n  RECOMMENDATION: Record more training data and retrain.")
        print("  (Current: 30 sequences per action)")
        print("  (Recommended: 50-100 sequences per action for stable learning)")

    return history


if __name__ == "__main__":
    train()
