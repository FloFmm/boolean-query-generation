#!/bin/bash
# setup_conda_env.sh
# Creates a Conda environment in /tmp/mycondaenv and installs dependencies

set -e

ENV_PATH="/tmp/mycondaenv"
PYTHON_VERSION=3.9

# make sure conda is available
if ! command -v conda &> /dev/null; then
    echo "conda not found. Please load or install conda first."
    exit 1
fi

echo "Removing existing conda environment at $ENV_PATH (if any)..."
conda remove --prefix "$ENV_PATH" --all -y || true

echo "Creating conda environment at $ENV_PATH..."
conda create --prefix "$ENV_PATH" python=$PYTHON_VERSION -y

echo "Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r /data/horse/ws/flml293c-master-thesis/boolean-query-generation/requirements.txt

echo "Conda environment ready at $ENV_PATH"
echo "To activate it later:"
echo "  source \$(conda info --base)/etc/profile.d/conda.sh"
echo "  conda activate $ENV_PATH"
