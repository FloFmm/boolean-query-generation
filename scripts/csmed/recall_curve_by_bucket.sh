#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=florian_maurus.mueller@mailbox.tu-dresden.de

# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# chmod +x ./boolean-query-generation/scripts/update_repos.sh
# ./boolean-query-generation/scripts/update_repos.sh

CSMED_PY=/data/horse/ws/flml293c-master-thesis/systematic-review-datasets/csmed_conda/bin/python

cd systematic-review-datasets
$CSMED_PY csmed/experiments/statistics_from_rankings.py
cd ../boolean-query-generation
$CSMED_PY app/visualization/recall_curve_by_bucket.py

# /data/horse/ws/flml293c-master-thesis/systematic-review-datasets/csmed_conda/bin/python csmed/experiments/csmed_cochrane_retrieval.py
