"""Script to verify that all training and test datasets load correctly with tab separator.

Prints the shape, columns, and first few rows (head) of each file.
"""

import sys
import gc
from pathlib import Path
import pandas as pd

# Add project root to sys.path to allow running directly or as a module
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import ALL_FILES


def inspect_file(name: str, path: Path, nrows_for_head: int = 3):
    print("=" * 80)
    print(f"Dataset: {name}")
    print(f"Path:    {path}")

    if not path.exists():
        print(f"ERROR: File not found at {path}")
        return

    # Check file size
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"File Size: {size_mb:.2f} MB")

    # Load with tab separator
    # To be memory efficient, we can get total rows and inspect head
    try:
        df = pd.read_csv(path, sep="\t", low_memory=False)
        print(f"Shape:   {df.shape} (rows: {df.shape[0]:,}, columns: {df.shape[1]})")
        print(f"Columns: {list(df.columns)}")
        print("\nHead (first few rows):")
        print(df.head(nrows_for_head))
        print("\nColumn Data Types & Non-Null Counts:")
        print(df.info(memory_usage="deep"))
        
        # Clean up memory
        del df
        gc.collect()
    except Exception as e:
        print(f"ERROR loading {name}: {e}")
    print("=" * 80 + "\n")


def main():
    print("\nVerifying datasets for Business Entity Resolution ML Challenge...\n")
    for name, path in ALL_FILES.items():
        inspect_file(name, path)


if __name__ == "__main__":
    main()
