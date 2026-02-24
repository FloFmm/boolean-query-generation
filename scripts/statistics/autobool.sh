#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=florian_maurus.mueller@mailbox.tu-dresden.de

# Path where the new conda env should live
set -e

ENV_PATH=/data/horse/ws/flml293c-master-thesis/systematic-review-dataset/autobool_conda
ENV_PY=/data/horse/ws/flml293c-master-thesis/systematic-review-dataset/autobool_conda/bin/python
PYTHON_VERSION=3.10
# Load conda (adjust if your cluster uses a different module name)
source "$(conda info --base)/etc/profile.d/conda.sh"
# Create env only if it doesn't exist
if [ ! -f "$ENV_PATH/bin/python" ]; then
echo "Creating conda environment at $ENV_PATH"
conda create --prefix "$ENV_PATH" python=$PYTHON_VERSION -y
conda activate "$ENV_PATH"

    # Install minimal dependencies
    pip install --no-cache-dir \
        "transformers>=4.40" \
        "tokenizers>=0.15" \
        accelerate \
        huggingface_hub \
        torch


else
echo "Using existing environment at $ENV_PATH"
conda activate "$ENV_PATH"
fi

# Go to project directory

cd boolean-query-generation

# Run script using env python
$ENV_PY -m app.statistics.autobool
