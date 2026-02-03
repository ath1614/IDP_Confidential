#!/bin/bash

# run_ocr.sh
# Usage: ./run_ocr.sh <input_folder>

if [ -z "$1" ]; then
    echo "Usage: $0 <input_folder>"
    exit 1
fi

INPUT_FOLDER=$1

# Activate venv (assuming it's in the project root or scripts folder)
# Adjust path as needed based on where setup_vm.sh created it
if [ -d "../venv" ]; then
    source ../venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "Running OCR Mode on $INPUT_FOLDER..."
python3 ../app.py --mode ocr --input "$INPUT_FOLDER"
