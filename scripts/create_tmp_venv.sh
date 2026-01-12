#!/bin/bash
# setup_venv.sh
# Creates a Python virtual environment in /tmp/myvenv and installs dependencies

# exit if any command fails
set -e

# path to virtual environment
VENV_PATH="/tmp/myvenv"

# Python executable (adjust if needed, e.g., python3.11)
PYTHON=python3

# remove existing venv if any
if [ -d "$VENV_PATH" ]; then
    echo "Removing existing virtual environment at $VENV_PATH"
    rm -rf "$VENV_PATH"
fi

# create venv
echo "Creating virtual environment at $VENV_PATH..."
$PYTHON -m venv "$VENV_PATH"

# activate venv
source "$VENV_PATH/bin/activate"

# upgrade pip
pip install --upgrade pip

# install requirements (optional)
pip install -r /data/horse/ws/flml293c-master-thesis/boolean-query-generation/requirements_disjunctive_dt.txt

echo "Virtual environment ready at $VENV_PATH"
echo "To activate it: source $VENV_PATH/bin/activate"