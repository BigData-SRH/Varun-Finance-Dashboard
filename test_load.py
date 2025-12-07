import traceback
import os
from utils.data_fetcher import load_csv_data

path = os.path.join('data', 'indices', 'NSEI.csv')
print(f'About to load: {path}')
print(f'Type: {type(path)}')
print(f'Exists: {os.path.exists(path)}')

try:
    result = load_csv_data(path)
    print(f'Success: {result is not None}')
    if result is not None:
        print(f'Rows: {len(result)}')
except Exception as e:
    print(f'Error: {e}')
    traceback.print_exc()
