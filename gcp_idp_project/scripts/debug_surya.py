import sys
import os
import importlib.util

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")

try:
    import surya
    print(f"\nSurya Location: {surya.__file__}")
    surya_dir = os.path.dirname(surya.__file__)
    
    print("\nSearching for 'def run_ocr' in surya directory...")
    found = False
    for root, dirs, files in os.walk(surya_dir):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                        if "def run_ocr" in content:
                            print(f"\n[FOUND] 'def run_ocr' in: {path}")
                            # Calculate relative import path
                            rel_path = os.path.relpath(path, os.path.dirname(surya_dir))
                            module_path = rel_path.replace(os.path.sep, ".").replace(".py", "")
                            print(f"       Likely import: from {module_path} import run_ocr")
                            found = True
                except Exception as e:
                    pass
    
    if not found:
        print("\n[WARNING] Could not find 'def run_ocr' in any file.")

    print("\nAttempting Imports with 'requests' check:")
    try:
        import requests
        print(" [SUCCESS] import requests")
    except ImportError:
        print(" [FAILED]  import requests (Please install 'requests')")

except ImportError:
    print("\n[CRITICAL] Could not import 'surya' package at all.")
