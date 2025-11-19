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

# clone/update code repos
update_repo git@github.com:FloFmm/boolean-query-generation.git boolean-query-generation
update_repo git@github.com:FloFmm/systematic-review-datasets.git systematic-review-datasets
update_repo git@github.com:FloFmm/CSMeD-baselines.git CSMeD-baselines

# clone/update data repos
update_repo https://github.com/ielab/SIGIR2017-SysRev-Collection.git SIGIR2017-SysRev-Collection
update_repo https://github.com/CLEF-TAR/tar.git tar
update_repo https://github.com/Amal-Alharbi/Systematic_Reviews_Update.git Systematic_Reviews_Update