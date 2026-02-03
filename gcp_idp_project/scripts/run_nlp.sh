#!/bin/bash

# run_nlp.sh
# Usage: ./run_nlp.sh <input_folder>

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

echo "Running NLP Mode on $INPUT_FOLDER..."
# Run app.py from Project Root
cd "$PROJECT_ROOT"
python3 app.py --mode nlp --input "$INPUT_FOLDER"
