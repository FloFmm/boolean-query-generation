#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=florian_maurus.mueller@mailbox.tu-dresden.de

# chmod +x ./boolean-query-generation/scripts/update_repos.sh
# ./boolean-query-generation/scripts/update_repos.sh

CSMED_PY=/data/horse/ws/flml293c-master-thesis/systematic-review-datasets/csmed_conda/bin/python

cd boolean-query-generation
$CSMED_PY -m app.dataset.analyze_datasets

# /data/horse/ws/flml293c-master-thesis/systematic-review-datasets/csmed_conda/bin/python -m app.dataset.build_bag_of_words_csmed