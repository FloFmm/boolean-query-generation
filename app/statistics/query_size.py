import json
import re
from collections import defaultdict
from app.config.config import TRAIN_REVIEWS
from app.dataset.utils import review_id_to_dataset, get_dataset_details
from collections import Counter
import statistics

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

def calc_manual_query_size():
    qg_results_path = "data/statistics/optuna/best1/lc=True,maxdf=0.5,mesh=True,ma=True,mindf=100,rw=True,rmn=True,rmp=True,d=503679/boot,=True,cw=0.2,dc=2,maxd=4,maxf=0.5,mof=10,maxs=None,midre=0.001,midrs=0.001,mins=3,mwfl=0.0002,ne=50,pfs=1.1,rmf=0.9,rmidr=0.9,rweight=1.5,k=1.5,tkoc=500/cf=0.002,cb=1.8,mh_noexp=False,mro=0.01,mrp=0.01,mto=0.12,pb=0.6,te=False,tiab=False/qg_results.jsonl"

    dataset_details = get_dataset_details()

    def fmt(p, r):
        return f"(precision={p:.4f}, recall={r:.4f})"

    count = defaultdict(list)
    count_and = defaultdict(list)
    count_or = defaultdict(list)
    count_not = defaultdict(list)
    for query_id in dataset_details.keys():
        if query_id in TRAIN_REVIEWS:
            continue
        dataset_name,_,_ = review_id_to_dataset(review_id=query_id)
        if dataset_name == "tar2017":
            dataset_name = "tar2018"
            
        c_final = None
        c_and_final = None
        c_or_final = None
        c_not_final = None
        for accepted in ["pubmed", "medline", "central"]:#, " search strateg"]:
            for s_name in sorted(dataset_details[query_id]["search_strategy"].keys()):
                if "imaging" in s_name.lower():
                    continue
                strategy = dataset_details[query_id]["search_strategy"][s_name].lower()
                if accepted not in s_name.lower() or "search" not in s_name.lower() or "strategy" not in s_name.lower(): 
                    continue
                cs = set()
                cs_and = set()
                cs_or = set()
                cs_not = set()
                for s_part in strategy.lower().split("search strateg"):
                    invalid = False
                    c_or = 0
                    for m in re.finditer(r"or/(\d+)[\-\u2010\u2013](\d+)", s_part, flags=re.IGNORECASE):
                        end_pos = m.end()
                        if end_pos < len(s_part) and s_part[end_pos] != " ":
                            invalid = True
                            break
                    pairs = re.findall(r"or/(\d+)[\-\u2010\u2013](\d+)", s_part, flags=re.IGNORECASE)
                    for s, e in pairs:
                        c_or += (int(e)-int(s))
                        if int(e) > 99 or int(s) >= int(e):
                            invalid = True
                            break
                        
                    if invalid or numbers_repeat_majority(s_part, threshold=0.7):
                        cs = set()
                        cs_and = set()
                        cs_or = set()
                        cs_not = set()
                        break

                    c_and = s_part.lower().count(" and ")
                    c_and += s_part.lower().count(" near ")
                    c_and += len(re.findall(r"\badj(\d+|\s)", s_part, flags=re.IGNORECASE))

                    
                    c_or += s_part.lower().count(" or ")
                    c_not = s_part.lower().count(" not ")

                    c = c_or + c_and + c_not
                    cs.add(c)
                    cs_and.add(c_and)
                    cs_or.add(c_or)
                    cs_not.add(c_not)
                if cs:
                    c_final = max(cs)
                    c_and_final = max(cs_and)
                    c_or_final = max(cs_or)
                    c_not_final = max(cs_not)
                    break
            if c_final:
                break

        if c_final is not None:
            count[dataset_name].append(c_final)
            count_and[dataset_name].append(c_and_final)
            count_or[dataset_name].append(c_or_final)
            count_not[dataset_name].append(c_not_final)
            if c_final > 100:
                print(dataset_name, query_id, c_final)

    print(count)
    result = {}
    for dataset_name, cs in count.items():
        filtered_cs = [x for x in cs if x > 2]
        filtered_and = [x for x in count_and[dataset_name] if x is not None]
        filtered_or = [x for x in count_or[dataset_name] if x is not None]
        filtered_not = [x for x in count_not[dataset_name] if x is not None]
        
        # Helper function to calculate average and std dev
        def calc_stats(values):
            if not values:
                return {"avg": 0, "max": 0, "min": 0, "std": 0}
            avg = sum(values) / len(values)
            std = statistics.stdev(values) if len(values) > 1 else 0
            return {
                "avg": avg,
                "max": max(values),
                "min": min(values),
                "std": std,
            }
        
        result[dataset_name] = {
            "\#ANDs": calc_stats(filtered_and),
            "\#ORs": calc_stats(filtered_or),
            "\#NOTs": calc_stats(filtered_not),
            "\#Ops": calc_stats(filtered_cs),
        }
        print(
            dataset_name,
            f"{sum(filtered_cs)/len(filtered_cs):.1f} logical opertors, max={max(filtered_cs)}, min={min(filtered_cs)}",
            f"AND avg={sum(filtered_and)/len(filtered_and):.1f}, max={max(filtered_and)}, min={min(filtered_and)}",
            f"OR avg={sum(filtered_or)/len(filtered_or):.1f}, max={max(filtered_or)}, min={min(filtered_or)}",
            f"NOT avg={sum(filtered_not)/len(filtered_not):.1f}, max={max(filtered_not)}, min={min(filtered_not)}",
        )
    return result
        
        