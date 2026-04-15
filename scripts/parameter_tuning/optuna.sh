#!/bin/bash
#SBATCH --nodes=50
#SBATCH --tasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=188130
#SBATCH --mail-type=end
#SBATCH --mail-user=florian_maurus.mueller@mailbox.tu-dresden.de

# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# Otherwise NumPy may spawn threads silently → catastrophic oversubscription.

CSMED_PY=[Path to your workspace]/systematic-review-datasets/csmed_conda/bin/python

cd boolean-query-generation
srun $CSMED_PY -m app.parameter_tuning.optuna --opt_beta 3.0
