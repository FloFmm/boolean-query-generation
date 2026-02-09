import json
from app.dataset.utils import get_dataset_details

qg_results_path = "data/statistics/optuna/best/lc=True,maxdf=0.5,mesh=True,ma=True,mindf=100,rw=True,rmn=True,rmp=True,d=503679/boot,=True,cw=0.2,dc=2,maxd=4,maxf=0.5,mof=10,maxs=None,midre=0.001,midrs=0.001,mins=3,mwfl=0.0002,ne=50,pfs=1.1,rmf=0.9,rmidr=0.9,rweight=1.5,k=1.5,tkoc=500/cf=0.002,cb=1.8,mh_noexp=False,mro=0.01,mrp=0.01,mto=0.12,pb=0.6,te=False,tiab=False/qg_results.jsonl"

dataset_details = get_dataset_details()
def fmt(p, r):
    return f"(precision={p:.4f}, recall={r:.4f})"

with_search_strategy = 0
without_search_strategy = 0
with open(qg_results_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        data = json.loads(line)
        query_id = data["query_id"]
        
        title = dataset_details[query_id]["title"]

        print(f"Query ID: {query_id}")
        print(f"Title   : {title}\n")

        print("PubMed Query:")
        print(data["pubmed_query"], "\n")

        print(
            "PubMed :", fmt(data["pubmed_precision"], data["pubmed_recall"])
        )
        print(
            "Subset :", fmt(data["subset_precision"], data["subset_recall"])
        )
        print(
            "Pseudo :", fmt(data["pseudo_precision"], data["pseudo_recall"])
        )
        print("-" * 40)