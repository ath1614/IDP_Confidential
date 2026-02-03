import sys
import os
import importlib.util

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")

try:
    import surya
    print(f"\nSurya Location: {surya.__file__}")
    surya_dir = os.path.dirname(surya.__file__)
    
    print("\nSurya Directory Contents:")
    try:
        files = os.listdir(surya_dir)
        for f in files:
            print(f" - {f}")
    except Exception as e:
        print(f"Error listing directory: {e}")

    print("\nAttempting Imports:")
    
    modules_to_try = [
        "surya.ocr",
        "surya.recognition",
        "surya.detection",
        "surya.model.recognition.model",
        "surya.model.detection.model",
        "surya.run_ocr",
        "surya.api"
    ]
    
    for mod in modules_to_try:
        try:
            importlib.import_module(mod)
            print(f" [SUCCESS] import {mod}")
            if mod == "surya.ocr":
                from surya.ocr import run_ocr
                print(f"   -> run_ocr found: {run_ocr}")
        except ImportError as e:
            print(f" [FAILED]  import {mod}: {e}")
        except Exception as e:
            print(f" [ERROR]   import {mod}: {e}")

except ImportError:
    print("\n[CRITICAL] Could not import 'surya' package at all.")
