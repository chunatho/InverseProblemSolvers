#!/bin/bash

# Bash script to set up and activate a virtual environment

module load python/3.12.2
module load gcc/14.2.0

# Step 1: Create virtual environment
VENV_NAME=".venv"
PYTHON_BIN="python"

echo " Creating virtual environment in '$VENV_NAME'..."
$PYTHON_BIN -m venv $VENV_NAME

# Step 2: Activate the virtual environment
echo "Activating virtual environment..."
source $VENV_NAME/bin/activate

# Step 3: Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Step 4: Install specific packages (add or modify this line as needed)
pip install matplotlib Dualperspective numba scipy

echo "Virtual environment setup complete!"
echo "To activate, run: source $VENV_NAME/bin/activate"
echo "then to solve the problems use ./run_all_inversions.sh"

