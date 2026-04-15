#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=florian_maurus.mueller@mailbox.tu-dresden.de

CSMED_PY=[Path to your workspace]/systematic-review-datasets/csmed_conda/bin/python

cd boolean-query-generation
$CSMED_PY -m app.statistics.feature_replacement_map