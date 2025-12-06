import sys
import numpy as np
from pathlib import Path

# Add backend to path so we can import utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.utils.normalization import normalize_landmarks


def test_normalization_basic():
    """
    Test that normalization correctly translates Wrist to origin
    and scales the reference bone to distance 1.0
    """
    # Create dummy landmarks: 21 points with arbitrary coordinates
    # We'll make a simple configuration where Wrist is at (1, 1, 1)
    # and Middle MCP is at (2, 1, 1) so distance is 1.0 before normalization
    landmarks = np.zeros((21, 3), dtype=np.float32)
    
    # Wrist at Index 0: (1, 1, 1)
    landmarks[0] = [1.0, 1.0, 1.0]
    
    # Middle MCP at Index 9: (2, 1, 1) - distance from wrist is 1.0
    landmarks[9] = [2.0, 1.0, 1.0]
    
    # Fill other landmarks with random values for completeness
    for i in range(1, 21):
        if i != 9:
            landmarks[i] = [1.5 + i * 0.1, 1.0 + i * 0.05, 1.0 + i * 0.02]
    
    # Normalize
    normalized = normalize_landmarks(landmarks)
    
    # Reshape to (21, 3) for easier assertion
    normalized_reshaped = normalized.reshape((21, 3))
    
    # Assert 1: Wrist coordinates should be exactly [0, 0, 0]
    wrist_normalized = normalized_reshaped[0]
    assert np.allclose(wrist_normalized, [0, 0, 0]), \
        f"Wrist should be at [0, 0, 0], got {wrist_normalized}"
    print("✓ Test 1 passed: Wrist is at origin [0, 0, 0]")
    
    # Assert 2: Distance between Wrist (Index 0) and Middle MCP (Index 9) should be 1.0
    middle_mcp_normalized = normalized_reshaped[9]
    distance = np.linalg.norm(middle_mcp_normalized - wrist_normalized)
    assert np.isclose(distance, 1.0), \
        f"Reference bone distance should be 1.0, got {distance}"
    print("✓ Test 2 passed: Reference bone distance (Index 0 to Index 9) is 1.0")


def test_normalization_with_scale_variance():
    """
    Test that normalization makes gestures scale-invariant.
    Two identical hand gestures at different scales should produce the same normalized output.
    """
    # Original gesture at scale 1
    landmarks_1x = np.zeros((21, 3), dtype=np.float32)
    landmarks_1x[0] = [0, 0, 0]  # Wrist
    landmarks_1x[9] = [1, 0, 0]  # Middle MCP
    for i in range(1, 21):
        if i != 9:
            landmarks_1x[i] = [0.5 + i * 0.05, 0.1, 0.2]
    
    # Same gesture at 2x scale
    landmarks_2x = landmarks_1x * 2
    landmarks_2x[0] = [0, 0, 0]  # Keep wrist at same position
    landmarks_2x[9] = [2, 0, 0]  # Middle MCP at 2x distance
    
    # Normalize both
    normalized_1x = normalize_landmarks(landmarks_1x)
    normalized_2x = normalize_landmarks(landmarks_2x)
    
    # They should be equal (scale-invariant)
    assert np.allclose(normalized_1x, normalized_2x), \
        "Normalized outputs should be identical for scale-variant inputs"
    print("✓ Test 3 passed: Normalization is scale-invariant")


def test_output_shape():
    """
    Test that output is correctly flattened to shape (63,)
    """
    landmarks = np.random.rand(21, 3).astype(np.float32)
    landmarks[0] = [0, 0, 0]  # Wrist at origin
    landmarks[9] = [1, 0, 0]  # Reference bone distance
    
    normalized = normalize_landmarks(landmarks)
    
    assert normalized.shape == (63,), \
        f"Expected shape (63,), got {normalized.shape}"
    print("✓ Test 4 passed: Output shape is (63,)")


def test_invalid_landmarks():
    """
    Test that appropriate errors are raised for invalid inputs
    """
    # Test with wrong number of landmarks
    try:
        bad_landmarks = np.random.rand(20, 3)  # Should be 21
        normalize_landmarks(bad_landmarks)
        assert False, "Should raise ValueError for wrong number of landmarks"
    except ValueError as e:
        print(f"✓ Test 5 passed: Correctly raises error for invalid landmark count: {e}")
    
    # Test with wrong number of coordinates
    try:
        bad_landmarks = np.random.rand(21, 2)  # Should be 3
        normalize_landmarks(bad_landmarks)
        assert False, "Should raise ValueError for wrong coordinate count"
    except ValueError as e:
        print(f"✓ Test 6 passed: Correctly raises error for invalid coordinate count: {e}")


def run_all_tests():
    """Run all test cases"""
    print("=" * 60)
    print("Running Normalization Tests")
    print("=" * 60)
    
    try:
        test_normalization_basic()
        test_normalization_with_scale_variance()
        test_output_shape()
        test_invalid_landmarks()
        
        print("=" * 60)
        print("All Tests Passed ✓")
        print("=" * 60)
        return True
    except AssertionError as e:
        print(f"\n❌ Test Failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
