# #!/bin/bash
# #SBATCH --nodes=1
# #SBATCH --tasks-per-node=1
# #SBATCH --cpus-per-task=1
# #SBATCH --time=00:10:00
# #SBATCH --mail-type=end
# #SBATCH --mail-user=florian_maurus.mueller@mailbox.tu-dresden.de

# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# chmod +x ./boolean-query-generation/scripts/update_repos.sh
# ./boolean-query-generation/scripts/update_repos.sh

CSMED_PY=/data/horse/ws/flml293c-master-thesis/systematic-review-datasets/csmed_conda/bin/python

cd boolean-query-generation
$CSMED_PY -m app.visualization.visualize_results
