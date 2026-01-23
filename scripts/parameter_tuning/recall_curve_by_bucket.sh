#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=florian_maurus.mueller@mailbox.tu-dresden.de

CSMED_PY=/data/horse/ws/flml293c-master-thesis/systematic-review-datasets/csmed_conda/bin/python

cd systematic-review-datasets
$CSMED_PY -m csmed.experiments.statistics_from_rankings
cd ../boolean-query-generation
$CSMED_PY -m app.visualization.recall_curve_by_bucket

# /data/horse/ws/flml293c-master-thesis/systematic-review-datasets/csmed_conda/bin/python csmed/experiments/csmed_cochrane_retrieval.py
