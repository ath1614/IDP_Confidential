#!/bin/bash

# setup_vm.sh
# Sets up the GCP VM for the IDP system.
# Assumes Ubuntu/Debian based system.

set -e

echo "Starting VM Setup..."

# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install system dependencies
# python3-venv, git, and libraries for opencv/pillow if needed
sudo apt-get install -y python3-pip python3-venv git libgl1-mesa-glx libglib2.0-0

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing Python dependencies..."
pip install -r ../requirements.txt

# Create necessary directories if they don't exist
mkdir -p ../ingest/apar
mkdir -p ../ingest/disciplinary/brief_background
mkdir -p ../ingest/disciplinary/po_brief
mkdir -p ../ingest/disciplinary/co_brief
mkdir -p ../ingest/disciplinary/io_report
mkdir -p ../output
mkdir -p ../logs
mkdir -p ../models

echo "Setup Complete. Activate environment with 'source scripts/venv/bin/activate' (if created in scripts) or adjusted path."
echo "Note: Models will be downloaded on first run or should be manually placed in models/."
