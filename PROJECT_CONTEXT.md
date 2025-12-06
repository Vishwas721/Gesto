# PRODUCT REQUIREMENTS DOCUMENT (PRD)
# Project Name: Gesto
# Version: 1.0 (MVP)

## 1. Executive Summary
Gesto is a contactless coding interface that translates custom hand gestures into syntactic code blocks in real-time. It targets developers with RSI (Repetitive Strain Injury) or motor impairments. It consists of a Python-based Computer Vision backend and a VS Code Extension frontend.

## 2. System Architecture
The system follows a Client-Server model over localhost TCP.
1. **The Eye (Backend):** Python 3.9+. Uses MediaPipe for hand tracking and an LSTM (Long Short-Term Memory) neural network to classify temporal gestures.
2. **The Transport:** Asyncio TCP Server. Uses "Length-Prefix Framing" (4-byte Big Endian header) to send JSON payloads.
3. **The Hand (Frontend):** VS Code Extension (TypeScript). Connects via TCP, receives JSON commands (e.g., `{"command": "type", "text": "for i in range():"}`), and edits the active document.

## 3. Core Technical Constraints
- **Latency:** End-to-end latency must be < 50ms. Nagle's Algorithm must be disabled (`TCP_NODELAY`).
- **Normalization:** Raw coordinates must be normalized using "Bone-Metric Scaling" (Euclidean distance between Wrist and Middle-MCP = 1.0) to ensure scale invariance.
- **Idle Detection:** Must implement a variance-based gate to prevent noise from triggering the LSTM.

## 4. MVP Feature Set
- **Gesture A:** "For Loop" (Circular motion) -> types `for i in range():`
- **Gesture B:** "Function" (Box shape) -> types `def function_name():`
- **Gesture C:** "Idle" (Random movement) -> No output.

## 5. Technology Stack
- **Backend:** Python, OpenCV, MediaPipe, NumPy, TensorFlow/Keras (for LSTM), Asyncio.
- **Frontend:** TypeScript, VS Code API.