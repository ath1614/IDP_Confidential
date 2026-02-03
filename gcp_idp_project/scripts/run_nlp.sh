#!/bin/bash

# run_nlp.sh
# Usage: ./run_nlp.sh <input_folder>

if [ -z "$1" ]; then
    echo "Usage: $0 <input_folder>"
    exit 1
fi

INPUT_FOLDER=$1

# Activate venv
if [ -d "../venv" ]; then
    source ../venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "Running NLP Mode on $INPUT_FOLDER..."
python3 ../app.py --mode nlp --input "$INPUT_FOLDER"
