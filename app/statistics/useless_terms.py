
from collections import defaultdict
import json
import os

from app.config.config import CURRENT_BEST, CURRENT_BEST_RUN_FOLDER
from app.dataset.utils import find_qg_results_file, get_dataset_details, get_qg_results, get_paper_query_examples, review_id_to_dataset
from app.pubmed.retrieval import evaluate_query
from app.helper.helper import f_beta
from app.visualization.helper import split_query_into_words

def find_useless_terms(query, end_year, positives, output_path=None):
    # Split by operators and parentheses, but do not capture them.
    tokens = split_query_into_words(query)
    token_count = len(tokens)
    tokens = set(tokens)
    
    useless_terms = []
    
    try:
        base_precision, base_recall, retrieved_count, TP = evaluate_query(
            query,
            positives,
            end_year=end_year,
            max_retrieved=1_000_000,
        )
        base_f_beta = f_beta(base_precision, base_recall, beta=50.0)
    except Exception as e:
        print(f"Error evaluating base query: {e}")
        return []

    print(f"Base F-beta: {base_f_beta}")

    for token in tokens:
        if not token.strip(" ()"):
            continue
        
        # Find all occurrences of this token
        occurrence_index = 0
        search_start = 0
        
        while True:
            # Find the next occurrence of the token
            pos = query.find(token, search_start)
            if pos == -1:
                break
            
            # Remove only this specific occurrence
            new_query = query[:pos] + query[pos + len(token):]
            search_start = pos + len(token)  # Continue searching after the current token
            
            # Check if performance is affected
            try:
                precision, recall, retrieved_count, TP = evaluate_query(
                    new_query,
                    positives,
                    end_year=end_year,
                    max_retrieved=1_000_000,
                )
                current_f_beta = f_beta(precision, recall, beta=50.0)
                
                # If performance is same or better, it's useless (or harmful)
                if current_f_beta >= base_f_beta:
                    print(f"Found useless term: '{token}' at occurrence {occurrence_index + 1} (position {pos}) (New F-beta: {current_f_beta})")
                    useless_terms.append({
                        "term": token,
                        "occurrence_index": occurrence_index,
                        "num_occurrences": query.count(token),
                        "position": pos,
                        "f_beta": current_f_beta,
                        "f_beta_increase": current_f_beta - base_f_beta,
                        "precision_increase": precision - base_precision,
                        "recall_increase": recall - base_recall,
                    })
                    print(useless_terms[-1])
            except Exception as e:
                # If the query becomes invalid, we ignore this removal
                pass
            
            occurrence_index += 1

    useless_terms.sort(key=lambda x: (x["term"], x["occurrence_index"]))

    if output_path:
        import json
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(useless_terms, f, indent=4)
            
    return useless_terms, len(useless_terms), len(useless_terms)/token_count


if __name__ == "__main__":
    path = find_qg_results_file(
        CURRENT_BEST_RUN_FOLDER, top_k_type="cosine", betas_key="50"
    )
    out_path = f"data/examples/useless_terms_{CURRENT_BEST}"
    dataframe = get_qg_results(path, min_positive_threshold=50)
    dataset_details = get_dataset_details()
    query_ids = ["CD007394", "CD009579", "CD009579"]
    results = {paper_name: defaultdict(list) for paper_name in ["kusaCSMeDBridgingDataset2023", "my_query"]}
    for query_id in query_ids:
        _, _, end_year = review_id_to_dataset(query_id)
        positives = set(dataset_details[query_id]["positives"])
        manual_query = get_paper_query_examples(paper="kusaCSMeDBridgingDataset2023", query_id=query_id)["result"]
        my_query = dataframe[dataframe["query_id"] == query_id]["pubmed_query"].values[0]
        
        
        data, num_useless, ratio = find_useless_terms(manual_query, end_year, positives, output_path=f"{out_path}/useless_manual_{query_id}.json")
        results["kusaCSMeDBridgingDataset2023"]["ratios"].append(ratio)
        results["kusaCSMeDBridgingDataset2023"]["num_useless"].append(num_useless)
        print("manual useless term ratio:", ratio)
        data, num_useless, ratio = find_useless_terms(my_query, end_year, positives, output_path=f"{out_path}/useless_my_query_{query_id}.json")
        results["my_query"]["ratios"].append(ratio)
        results["my_query"]["num_useless"].append(num_useless)
        print("my query useless term ratio:", ratio)
    
    for paper_name, ratios in results.items():
        avg_ratio = sum(ratios["ratios"]) / len(ratios["ratios"]) if ratios["ratios"] else 0
        avg_num_useless = sum(ratios["num_useless"]) / len(ratios["num_useless"]) if ratios["num_useless"] else 0
        print(f"Average useless term ratio for {paper_name}: {avg_ratio:.2%}")
        results[paper_name] |= {
            "avg_ratio": avg_ratio,
            "avg_num_useless": avg_num_useless
        }
    
    with open(f"{out_path}/useless_term_stats.json", "w") as f:
        json.dump(results, f, indent=4)
        