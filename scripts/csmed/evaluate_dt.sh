#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --array=0-1500%512
#SBATCH --mail-type=end
#SBATCH --mail-user=florian_maurus.mueller@mailbox.tu-dresden.de
#SBATCH --output=logs/out_%A_%a.txt
#SBATCH --error=logs/err_%A_%a.txt

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

CSMED_PY=/data/horse/ws/flml293c-master-thesis/systematic-review-datasets/csmed_conda/bin/python

cd boolean-query-generation
echo "Running job index: $SLURM_ARRAY_TASK_ID"
$CSMED_PY -m app.experiments.evaluate_dt_csmed $SLURM_ARRAY_TASK_ID
# $CSMED_PY -m app.experiments.evaluate_pubmed_query
# $CSMED_PY -m app.tree_learning.disjunctive_dt

# /data/horse/ws/flml293c-master-thesis/systematic-review-datasets/csmed_conda/bin/python app/experiments/evaluate_dt_csmed.py
