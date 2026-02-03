import sys
import os
import surya

surya_base = os.path.dirname(surya.__file__)

def read_file(rel_path):
    path = os.path.join(surya_base, rel_path)
    print(f"\n\n{'='*20} {rel_path} {'='*20}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            print(f.read())
    except Exception as e:
        print(f"Error reading {path}: {e}")

# Read the CLI script to see how it imports and runs things
read_file("scripts/ocr_text.py")

# Also check models.py as it's in the root
read_file("models.py")

# Check detection loader
read_file("detection/loader.py")
