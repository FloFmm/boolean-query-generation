#!/bin/bash
set -e

DEST=/home/florian/Data/dev/boolean-query-generation/data
mkdir -p "$DEST/statistics"
mkdir -p "$DEST/statistics/optuna"
SOURCE=dataport1.hpc.tu-dresden.de:/data/horse/ws/flml293c-master-thesis/boolean-query-generation/data/
rsync -av --progress "$SOURCE/reports" "$DEST"
rsync -av --progress "$SOURCE/statistics/optuna/images" "$DEST/statistics/optuna"
rsync -av --progress "$SOURCE/examples/feature_replacement_map_*" "$DEST/examples"
rsync -av --progress "$SOURCE/examples/useless_terms_*" "$DEST/examples"
rsync -av --progress "$SOURCE/examples/autobool_results.jsonl" "$DEST/examples"
rsync -av --progress --exclude='*.pkl' --exclude='*.privatelock' --exclude='*.lock' $SOURCE/statistics/optuna/best* "$DEST/statistics/optuna"
rsync -av --progress --exclude='*.pkl' --exclude='*.privatelock' --exclude='*.lock' $SOURCE/statistics/optuna/evaluate_base_* "$DEST/statistics/optuna"
rsync -av --progress --exclude='*.pkl' --exclude='*.privatelock' --exclude='*.lock' $SOURCE/statistics/optuna/evaluate_best_* "$DEST/statistics/optuna"
rsync -av --progress --relative $SOURCE/./statistics/optuna/run*/*.db $DEST/
rsync -av --progress --relative dataport1.hpc.tu-dresden.de:/data/horse/ws/flml293c-master-thesis/slurm-*.out $DEST/slurm_logs/

DEST=/home/florian/Data/dev/systematic-review-datasets/data
mkdir -p "$DEST"
mkdir -p "$DEST/rankings"
SOURCE=dataport1.hpc.tu-dresden.de:/data/horse/ws/flml293c-master-thesis/systematic-review-datasets/data
rsync -av --progress "$SOURCE/bag_of_words" "$DEST"
rsync -av --progress "$SOURCE/external" "$DEST"
rsync -av --progress "$SOURCE/rankings/pubmedbert" "$DEST/rankings"
rsync -av --progress "$SOURCE/dataset_details" "$DEST"