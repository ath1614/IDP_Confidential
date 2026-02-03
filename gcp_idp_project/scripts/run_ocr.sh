#!/bin/bash

# run_ocr.sh
# Usage: ./run_ocr.sh <input_folder>

# Resolve script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -z "$1" ]; then
    echo "Usage: $0 <input_folder>"
    exit 1
fi

INPUT_FOLDER=$1

# Activate venv
if [ -d "$PROJECT_ROOT/venv" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
elif [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

echo "Running OCR Mode on $INPUT_FOLDER..."
# Run app.py from Project Root to ensure relative paths work if app.py uses them
cd "$PROJECT_ROOT"
python3 app.py --mode ocr --input "$INPUT_FOLDER"
