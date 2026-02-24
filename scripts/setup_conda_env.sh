#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=florian_maurus.mueller@mailbox.tu-dresden.de
#SBATCH --mem=188130

# Load Anaconda / Python module
ml release/23.04
ml Anaconda3/2022.05
ml GCC/11.3.0

# conda env set up
echo "=== Setting up environment ==="
cd systematic-review-datasets
if [ ! -d "./csmed_conda" ]; then
    conda create --prefix ./csmed_conda python=3.10 -y
fi

# Use the Python inside the environment explicitly (instead of conda activate ./csmed_conda)
CSMED_PY=/data/horse/ws/flml293c-master-thesis/systematic-review-datasets/csmed_conda/bin/python
$CSMED_PY -m pip install -r requirements.txt
