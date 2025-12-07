"""Test if the app can load data successfully"""
import sys
import traceback

try:
    print("Importing modules...")
    from utils import load_all_index_data, INDICES
    
    print("Loading index data...")
    index_data = load_all_index_data()
    
    print(f"\nResults:")
    print(f"Expected indices: {list(INDICES.keys())}")
    print(f"Loaded indices: {list(index_data.keys())}")
    print(f"Success: {len(index_data)} / {len(INDICES)}")
    
    for name, data in index_data.items():
        print(f"  {name}: {len(data)} rows, index type={type(data.index[0])}")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    traceback.print_exc()
    sys.exit(1)
