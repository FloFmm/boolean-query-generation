![Overview](data/overview_big_final_cropped.png)

*1. Dense Retrieval:* The pipeline begins by using a dense retrieval model to rank documents from a local corpus by relevance to the query. 
Based on their rank, documents are assigned pseudo-relevance labels and a weight indicating the confidence in that label.

*2. Learning the Tree Ensemble:* A modified Random Forest is trained on the precomputed #gls("bow") representations of the pseudo-labeled documents. 
The goal is to learn a set of decision trees that can effectively separate relevant from non-relevant documents.

*3. Rule Pruning and Variation:* All decision paths leading to a pseudo-relevant classification are extracted as Boolean rules. 
These rules are then rigorously pruned to simplify them, and variations are generated for each to create a diverse pool of candidate rules.

*4. Rule Selection and Query Generation:* Finally, a high-performing, low-cost subset of rules is selected from the candidate pool. 
The selection maximizes retrieval performance on the pseudo-labeled local corpus while penalizing query size and complexity. 
The chosen rules are combined with the #OR operator to form the final Boolean query, which is then evaluated on #pubmed.

# Documentation (WIP)
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