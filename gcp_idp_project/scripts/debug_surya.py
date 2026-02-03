import sys
import os
import importlib.util

try:
    import surya
    surya_dir = os.path.dirname(surya.__file__)
    print(f"Surya Base Dir: {surya_dir}")

    print("\n--- Recursive File List ---")
    for root, dirs, files in os.walk(surya_dir):
        rel_root = os.path.relpath(root, surya_dir)
        if rel_root == ".":
            rel_root = ""
        for f in files:
            if f.endswith(".py"):
                print(os.path.join(rel_root, f))
    
    print("\n--- Checking __init__.py ---")
    with open(os.path.join(surya_dir, "__init__.py"), "r") as f:
        print(f.read())

except ImportError:
    print("Could not import surya")
