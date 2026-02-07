import sys
import os
import json
from app.dataset.utils import dataset_details_path

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../..", "../systematic-review-datasets"
        )
    )
)
from csmed.experiments.csmed_cochrane_retrieval import (
    load_dataset,
)

qg_results_path = "data/statistics/optuna/best/lc=True,maxdf=0.5,mesh=True,ma=True,mindf=100,rw=True,rmn=True,rmp=True,d=503679/boot,=True,cw=0.2,dc=2,maxd=4,maxf=0.5,mof=10,maxs=None,midre=0.001,midrs=0.001,mins=3,mwfl=0.0002,ne=50,pfs=1.1,rmf=0.9,rmidr=0.9,rweight=1.5,k=1.5,tkoc=500/cf=0.002,cb=1.8,mh_noexp=False,mro=0.01,mrp=0.01,mto=0.12,pb=0.6,te=False,tiab=False/qg_results.jsonl"

dataset = load_dataset()
# dataset already loaded earlier
reviews = dataset["EVAL"] | dataset["TRAIN"]

dataset_path = dataset_details_path()
queryid_to_details = {
    query_id: review_data["dataset_details"]
    for query_id, review_data in reviews.items()
}
out_path = dataset_details_path()
out_path.parent.mkdir(parents=True, exist_ok=True)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(queryid_to_details, f, indent=2)