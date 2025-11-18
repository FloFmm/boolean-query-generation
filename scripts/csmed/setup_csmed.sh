# clone code repos (my forks)
echo "=== Cloning code repositories ==="
[ ! -d "systematic-review-datasets" ] && git clone git@github.com:FloFmm/systematic-review-datasets.git
[ ! -d "CSMeD-baselines" ] && git clone git@github.com:FloFmm/CSMeD-baselines.git

echo "=== Setting up environment ==="
cd systematic-review-datasets
if [ ! -d "./csmed" ]; then
    conda create --prefix ./csmed python=3.10
fi
conda activate csmed
pip install -r requirements.txt
pip install -r experiment_requirements.txt

# clone data repos
echo "=== Cloning data repositories ==="
[ ! -d "SIGIR2017-SysRev-Collection" ] && git clone https://github.com/ielab/SIGIR2017-SysRev-Collection.git
[ ! -d "tar" ] && git clone https://github.com/CLEF-TAR/tar.git
[ ! -d "Systematic_Reviews_Update" ] && git clone https://github.com/Amal-Alharbi/Systematic_Reviews_Update.git

# get review details from cochrane
echo "=== Converting datasets ==="
cd scripts
python convert_tar_dataset.py
python convert_sigir2017_dataset.py
python convert_sr_updates_dataset.py