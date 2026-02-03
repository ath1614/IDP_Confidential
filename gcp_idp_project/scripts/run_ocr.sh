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

# Check if a virtual environment is already active
if [ -z "$VIRTUAL_ENV" ]; then
    # No venv active, try to find and activate one
    if [ -d "$PROJECT_ROOT/venv" ]; then
        echo "Activating venv at $PROJECT_ROOT/venv"
        source "$PROJECT_ROOT/venv/bin/activate"
    elif [ -d "$SCRIPT_DIR/venv" ]; then
        echo "Activating venv at $SCRIPT_DIR/venv"
        source "$SCRIPT_DIR/venv/bin/activate"
    else
        echo "Warning: No virtual environment found. Running with system python."
    fi
else
    echo "Using active virtual environment: $VIRTUAL_ENV"
fi

echo "Python Executable: $(which python3)"
echo "Running OCR Mode on $INPUT_FOLDER..."

# Run app.py from Project Root
cd "$PROJECT_ROOT"
python3 app.py --mode ocr --input "$INPUT_FOLDER"
