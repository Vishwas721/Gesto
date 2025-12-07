import numpy as np
import os

# Path to your data
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data_collection', 'MP_Data')
action = 'for_loop' # Check one of your classes

try:
    # Try to load the first sequence, first frame
    file_path = os.path.join(DATA_PATH, action, '0', '0.npy')
    data = np.load(file_path)
    
    print("\n=== DATA INSPECTION ===")
    print(f"Shape: {data.shape} (Should be (63,))")
    print(f"Data Type: {data.dtype}")
    print(f"Min Value: {np.min(data)}")
    print(f"Max Value: {np.max(data)}")
    print(f"Mean Value: {np.mean(data)}")
    print("\nFirst 10 values:")
    print(data[:10])
    
    # Check for Zeros (Empty data)
    if np.all(data == 0):
        print("\n[CRITICAL ERROR] Data is all ZEROS. The recorder isn't seeing the hand!")
    
    # Check for NaNs (Broken Math)
    elif np.isnan(data).any():
        print("\n[CRITICAL ERROR] Data contains NaNs. The normalization math divided by zero.")
        
    else:
        print("\n[PASS] Data looks statistically valid.")

except Exception as e:
    print(f"\n[ERROR] Could not load file: {e}")