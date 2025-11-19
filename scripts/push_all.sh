#!/bin/bash

# Function to push changes in a repo
push_repo() {
    local folder_name=$1
    local commit_msg=$2

    if [ -d "$folder_name" ]; then
        echo "=== Pushing changes in $folder_name ==="
        cd "$folder_name" || { echo "Cannot enter $folder_name"; return; }

        # Check if there are changes
        if [[ -n $(git status --porcelain) ]]; then
            git add .
            git commit -m "$commit_msg"
            git push
        else
            echo "No changes to push in $folder_name"
        fi

        cd ..
    else
        echo "Folder $folder_name does not exist, skipping."
    fi
}

# Default commit message (can be overridden by command line argument)
COMMIT_MSG=${1:-"Update from HPC"}

# List of repos to push
REPOS=(
    "boolean-query-generation"
    "systematic-review-datasets"
    "CSMeD-baselines"
    "SIGIR2017-SysRev-Collection"
    "tar"
    "Systematic_Reviews_Update"
)

# Loop over all repos
for repo in "${REPOS[@]}"; do
    push_repo "$repo" "$COMMIT_MSG"
done
