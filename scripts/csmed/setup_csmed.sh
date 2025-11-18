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
update_repo git@github.com:FloFmm/systematic-review-datasets.git systematic-review-datasets
update_repo git@github.com:FloFmm/CSMeD-baselines.git CSMeD-baselines

# clone/update data repos
update_repo https://github.com/ielab/SIGIR2017-SysRev-Collection.git SIGIR2017-SysRev-Collection
update_repo https://github.com/CLEF-TAR/tar.git tar
update_repo https://github.com/Amal-Alharbi/Systematic_Reviews_Update.git Systematic_Reviews_Update

# conda env set up
echo "=== Setting up environment ==="
conda init bash
source ~/.bashrc
cd systematic-review-datasets
if [ ! -d "./csmed" ]; then
    conda create --prefix ./csmed python=3.10
fi
conda activate csmed
pip install -r requirements.txt
pip install -r experiment_requirements.txt

# get review details from cochrane
echo "=== Converting datasets ==="
cd scripts
python convert_tar_dataset.py
python convert_sigir2017_dataset.py
python convert_sr_updates_dataset.py