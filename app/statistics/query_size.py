import json
import re
from collections import defaultdict
from app.dataset.utils import review_id_to_dataset, get_dataset_details
from collections import Counter

def numbers_repeat_majority(s: str, threshold: float = 0.7) -> bool:
    # Step 1: find all numbers
    numbers = re.findall(r'\d+', s)
    
    if not numbers:
        return False  # no numbers in string
    
    # Step 2: count occurrences
    counts = Counter(numbers)
    
    # Step 3: count how many numbers appear 2 or more times
    repeated_count = sum(1 for n in counts if counts[n] >= 2)
    
    # Step 4: check if the proportion exceeds the threshold
    proportion = repeated_count / len(counts)
    
    return proportion > threshold

qg_results_path = "data/statistics/optuna/best/lc=True,maxdf=0.5,mesh=True,ma=True,mindf=100,rw=True,rmn=True,rmp=True,d=503679/boot,=True,cw=0.2,dc=2,maxd=4,maxf=0.5,mof=10,maxs=None,midre=0.001,midrs=0.001,mins=3,mwfl=0.0002,ne=50,pfs=1.1,rmf=0.9,rmidr=0.9,rweight=1.5,k=1.5,tkoc=500/cf=0.002,cb=1.8,mh_noexp=False,mro=0.01,mrp=0.01,mto=0.12,pb=0.6,te=False,tiab=False/qg_results.jsonl"

dataset_details = get_dataset_details()

def fmt(p, r):
    return f"(precision={p:.4f}, recall={r:.4f})"

with_search_strategy = 0
without_search_strategy = 0
count = defaultdict(list)
with open(qg_results_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        data = json.loads(line)
        query_id = data["query_id"]
        dataset_name,_,_ = review_id_to_dataset(review_id=query_id)
        if dataset_name == "tar2017":
            dataset_name = "tar2018"
        
        c_final = None
        for accepted in ["pubmed", "medline", "central"]:#, " search strateg"]:
            for s_name in sorted(dataset_details[query_id]["search_strategy"].keys()):
                if "imaging" in s_name.lower():
                    continue
                strategy = dataset_details[query_id]["search_strategy"][s_name].lower()
                if accepted not in s_name.lower() or "search" not in s_name.lower() or "strategy" not in s_name.lower(): 
                    continue
                cs = set()
                for s_part in strategy.lower().split("search strateg"):
                    c = 0
                    invalid = False
                    for m in re.finditer(r"or/(\d+)[\-\u2010\u2013](\d+)", s_part, flags=re.IGNORECASE):
                        end_pos = m.end()
                        if end_pos < len(s_part) and s_part[end_pos] != " ":
                            invalid = True
                            break
                    pairs = re.findall(r"or/(\d+)[\-\u2010\u2013](\d+)", s_part, flags=re.IGNORECASE)
                    for s, e in pairs:
                        c += (int(e)-int(s))
                        if int(e) > 99 or int(s) >= int(e):
                            invalid = True
                            break
                        
                    if invalid or numbers_repeat_majority(s_part, threshold=0.7):
                        cs = set()
                        break
                    
                    c += s_part.lower().count(" near ")
                    c += len(re.findall(r"\badj(\d+|\s)", s_part, flags=re.IGNORECASE))
                    c += s_part.lower().count(" or ") + s_part.lower().count(" and ") + s_part.lower().count(" not ") 
                    cs.add(c)
                if cs:
                    c_final = max(cs)
                    break
            if c_final:
                break
        
        
        if c_final is not None:
            count[dataset_name].append(c_final)
            if c_final > 100:
                print(dataset_name, query_id, c_final)
        
print(count)
for dataset_name, cs in count.items():
    filtered_cs = [x for x in cs if x > 2]
    print(dataset_name, f"{sum(filtered_cs)/len(filtered_cs):.1f} logical opertors, max={max(filtered_cs)}, min={min(filtered_cs)}")