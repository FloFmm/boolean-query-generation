## setup
```bash
cd /data/horse/ws/flml293c-master-thesis
git clone git@github.com:FloFmm/boolean-query-generation.git
sbatch -A p_scads_bquery boolean-query-generation/scripts/csmed/setup_csmed.sh
```

## Install Spacy Model
```bash
cd /data/horse/ws/flml293c-master-thesis/systematic-review-datasets/data/spacy
wget https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1.tar.gz -P /data/horse/ws/flml293c-master-thesis/systematic-review-datasets/data/spacy
tar -xzf en_core_web_lg-3.7.1.tar.gz
```