#!/bin/bash
#SBATCH --nodes=20
#SBATCH --tasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=188130
#SBATCH --mail-type=end
#SBATCH --mail-user=florian_maurus.mueller@mailbox.tu-dresden.de

CSMED_PY=[Path to your workspace]/systematic-review-datasets/csmed_conda/bin/python

cd boolean-query-generation
srun $CSMED_PY -m app.experiments.evaluate_base
