#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=florian_maurus.mueller@mailbox.tu-dresden.de

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Load Anaconda / Python module
ml release/23.04
ml Anaconda3/2022.05
ml GCC/11.3.0

chmod +x ./boolean-query-generation/scripts/csmed/update_repos.sh
./boolean-query-generation/scripts/csmed/update_repos.sh

# conda env set up
echo "=== Setting up environment ==="
cd systematic-review-datasets
if [ ! -d "./csmed_conda" ]; then
    conda create --prefix ./csmed_conda python=3.10 -y
fi

# Use the Python inside the environment explicitly (instead of conda activate ./csmed_conda)
CSMED_PY=/data/horse/ws/flml293c-master-thesis/systematic-review-datasets/csmed_conda/bin/python

$CSMED_PY -m pip install --upgrade pip setuptools wheel
pip install cython
pip install pystemmer==2.0.1 --no-build-isolation
# $CSMED_PY -m pip install pystemmer==3.0.0
# $CSMED_PY -m pip install --no-deps retriv~=0.2.3
$CSMED_PY -m pip install -r requirements.txt

# get review details from cochrane
playwright install
echo "=== Converting datasets ==="
cd scripts
$CSMED_PY convert_tar_dataset.py
$CSMED_PY convert_sigir2017_dataset.py
$CSMED_PY convert_sr_updates_dataset.py

cd ..
cd ..
cd CSMeD-baselines
$CSMED_PY experiments/csmed_cochrane/csmed_cochrane_retrieval.py