#!/bin/bash

# Function to clone or update a repo
update_repo() {
    local repo_url=$1
    local folder_name=$2
    if [ -d "$folder_name" ]; then
        echo "=== Updating $folder_name ==="
        cd "$folder_name"
        git pull
        cd ..
    else
        echo "=== Cloning $folder_name ==="
        git clone "$repo_url" "$folder_name"
    fi
}

# Load Anaconda / Python module
ml release/23.04
ml Anaconda3/2022.05
ml GCC/13.2.0

# clone/update code repos
update_repo git@github.com:FloFmm/systematic-review-datasets.git systematic-review-datasets
update_repo git@github.com:FloFmm/CSMeD-baselines.git CSMeD-baselines

# clone/update data repos
update_repo https://github.com/ielab/SIGIR2017-SysRev-Collection.git SIGIR2017-SysRev-Collection
update_repo https://github.com/CLEF-TAR/tar.git tar
update_repo https://github.com/Amal-Alharbi/Systematic_Reviews_Update.git Systematic_Reviews_Update

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
# $CSMED_PY -m pip install -r experiment_requirements.txt

# get review details from cochrane
playwright install
echo "=== Converting datasets ==="
cd scripts
# $CSMED_PY convert_tar_dataset.py
# $CSMED_PY convert_sigir2017_dataset.py
# $CSMED_PY convert_sr_updates_dataset.py

cd ..
cd ..
cd CSMeD-baselines
$CSMED_PY experiments/csmed_cochrane/csmed_cochrane_retrieval.py