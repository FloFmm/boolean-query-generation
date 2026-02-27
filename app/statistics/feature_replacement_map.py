import os
import re

from app.config.config import CURRENT_BEST, CURRENT_BEST_RUN_FOLDER
from app.dataset.utils import find_qg_results_file, get_dataset_details, get_paper_query_examples, get_qg_results, review_id_to_dataset
from app.pubmed.retrieval import evaluate_query
from app.helper.helper import f_beta
from app.visualization.helper import split_query_into_words

def find_good_term_subsitutions(query1, query2, end_year, positives, output_path=None):
    # Split by operators and parentheses, but do not capture them.
    # We also might get empty strings if there are adjacent delimiters (like parentheses), so we filter them.
    tokens_q1 = [t.strip() for t in re.split(r'\s+AND\s+|\s+OR\s+|\s+NOT\s+|\(|\)', query1) if t.strip()]
    tokens_q2 = [t.strip() for t in re.split(r'\s+AND\s+|\s+OR\s+|\s+NOT\s+|\(|\)', query2) if t.strip()]
    
    
    replacements1 = {}
    replacements2 = {}
    precision1, recall1, retrieved_count1, TP1 = evaluate_query(
        query1,
        positives,
        end_year=end_year,
    )
    best_f1 = f_beta(precision1, recall1, beta=50.0)
    precision2, recall2, retrieved_count2, TP2 = evaluate_query(
        query2,
        positives,
        end_year=end_year,
    )
    best_f2 = f_beta(precision2, recall2, beta=50.0)
    print("start", best_f1, best_f2)
    
    tokens_q1 = set(tokens_q1)
    tokens_q2 = set(tokens_q2)
    for token1 in tokens_q1:
        for token2 in tokens_q2:
            if not token1.strip(" ()") or not token2.strip(" ()"):
                continue
            new_query2 = query2.replace(token2, token1)
            new_query1 = query1.replace(token1, token2)
            try:
                precision, recall, retrieved_count, TP = evaluate_query(
                    new_query1,
                    positives,
                    end_year=end_year,
                )
            except Exception as e:
                continue
            f_beta_val1 = f_beta(precision, recall, beta=50.0)
            if token1 not in replacements1:
                replacements1[token1] = []
            replacements1[token1].append((token2, f_beta_val1-best_f1))
            
            try:
                precision, recall, retrieved_count, TP = evaluate_query(
                    new_query2,
                    positives,
                    end_year=end_year,
                )
            except Exception as e:
                continue
            f_beta_val2 = f_beta(precision, recall, beta=50.0)
            if token2 not in replacements2:
                replacements2[token2] = []
            replacements2[token2].append((token1, f_beta_val2-best_f2))
            if f_beta_val2 > best_f2:
                print("good replacement found:", token2, "->", token1, "with f_beta:", f_beta_val2)
                
            print(token1, f_beta_val1, "<->", token2, f_beta_val2)
            
    for token in replacements1:
        replacements1[token].sort(key=lambda x: x[1], reverse=True)
    for token in replacements2:
        replacements2[token].sort(key=lambda x: x[1], reverse=True)

    if output_path:
        import json
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"replacements1": replacements1, "replacements2": replacements2}, f, indent=4)
                
    return replacements1, replacements2

if __name__ == "__main__":
    # review_id = "CD007394"
    # query1 = "((invasive OR \"immunoenzyme techniques\"[mh] OR detect) AND (galactomannan OR \"aspergillosis/diagnosis\"[mh] OR \"antigens, fungal/blood\"[mh]) AND (sensor OR \"mannans/blood\"[mh] OR \"epidemiologic research design\"[mh] OR diagnosi)) OR ((sensor OR \"aspergillosis/diagnosis\"[mh]) AND (\"mannans\"[mh] OR detect OR assay) AND (aspergillosis OR galactomannan))"
    # query2 = "(invasive OR cryptococcosis OR Patient immunosuppressed OR zygomycosis OR minimally-invasive OR galactomannan OR immunocompromised OR 1²-d-glucan OR immuno-compromised OR beta-d-glucan OR detections OR minimally OR detecting OR detection OR immunocompromized OR Aspergillosis) AND (aspergillosis)"
    # review_id = "CD009579"
    # query1 = "(antigen AND \"schistosomiasis mansoni/diagnosis\"\[mh\]) OR ((indirect OR \"enzyme-linked immunosorbent assay/methods\"\[mh\] OR sensor OR reagent) AND (\"trematode infections\"\[mh\] OR antigen OR \"chromatography\"\[mh\]) AND (immunochromatographic OR malaria OR \"schistosomiasis\"\[mh\] OR \"helicobacter infections/diagnosis\"\[mh\]) AND (\"sensitivity and specificity\"\[mh\] OR \"schistosomiasis haematobia/diagnosis\"\[mh\] OR detect OR \"schistosomiasis/diagnosis\"\[mh\])) OR ((sensor OR \"enzyme-linked immunosorbent assay/methods\"\[mh\] OR \"sensitivity and specificity\"\[mh\] OR gut) AND (immunochromatographic OR \"reagent kits, diagnostic\"\[mh\] OR \"antigens, bacterial/analysis\"\[mh\] OR hpsa OR \"schistosomiasis\"\[mh\] OR dipstick) AND (strip OR gold OR detect) AND (stool OR \"parasitic diseases\"\[mh\]))"
    # query2 = "((anodic\[tiab\] AND antigen\*\[tiab\]) OR (cathodic\[tiab\] AND antigen\*\[tiab\]) OR \"Enzyme-Linked Immunosorbent Assay\"\[mh\] OR \"Immunoenzyme Techniques\"\[mh\] OR hematuria\[mh:noexp\] OR proteinuria\[mh\] OR leukocyturia\[tiab\] OR leucocyturia\[tiab\] OR hematuria\[tiab\] OR proteinuria\[tiab\] OR albuminuria\[tiab\] OR CCA\[tiab\] OR CAA\[tiab\] OR urinalysis\[tiab\] OR elisa\[tiab\] OR eia\[tiab\] OR \"Reagent Strips\"\[mh\] OR dipstick OR (reagent\[tiab\] AND strip\*\[tiab\]) OR (test\[tiab\] AND strip\*\[tiab\]) OR haemastix\[tiab\] OR \"schistosoma mansoni\"\[tiab\] OR \"schistosoma haematobium\"\[tiab\] OR Glycoproteins\[mh\] OR \"Antigens, Helminth\"\[mh\] OR \"Helminth Proteins\"\[mh\] OR \"Schistosoma haematobium\"\[mh\] OR \"Antibodies, Monoclonal\"\[mh\] OR \"Schistosoma mansoni\"\[mh\]) AND (schistosomiasis\[mh:noexp\] OR \"schistosomiasis haematobia\"\[mh:noexp\] OR \"schistosomiasis mansoni\"\[mh:noexp\] OR schistosomiasis\[tiab\] OR bilharzia\*\[tiab\]) NOT (animals\[mh:noexp\] NOT humans\[mh:noexp\] OR Letter OR \"Case Reports\")"    
    
    
    
    # best_replacements_q1, best_replacements_q2 = find_good_term_subsitutions(query1, query2, end_year, positives)
    # print("Best replacements for Query 1:", best_replacements_q1)
    # print("Best replacements for Query 2:", best_replacements_q2)
    
    
    path = find_qg_results_file(
        CURRENT_BEST_RUN_FOLDER, top_k_type="cosine", betas_key="50"
    )
    out_path = f"data/examples/feature_replacement_map_{CURRENT_BEST}"
    dataframe = get_qg_results(path, min_positive_threshold=50)
    
    query_id1 = "CD007394"
    query_id2 = "CD009579"
    semantic_query = get_paper_query_examples(paper="pourrezaSemanticdrivenBooleanQuery2023", query_id=query_id1)["result"]
    chatgpt_query = get_paper_query_examples(paper="wangCanChatGPTWrite2023", query_id=query_id1)["result"]
    chatgpt_query = get_paper_query_examples(paper="wangAutoBoolReinforcementLearningTrained2025", query_id=query_id1)["result"]
    manual_query = get_paper_query_examples(paper="kusaCSMeDBridgingDataset2023", query_id= query_id2)["result"]
    objective_query = get_paper_query_examples(paper="scellsComputationalApproachObjectively2020", query_id= query_id2)["result"]
    
    query1 = dataframe[dataframe["query_id"] == query_id1]["pubmed_query"].values[0]
    query2 = dataframe[dataframe["query_id"] == query_id2]["pubmed_query"].values[0]
    _, _, end_year1 = review_id_to_dataset(query_id1)
    _, _, end_year2 = review_id_to_dataset(query_id2)
    dataset_details = get_dataset_details()
    positives1 = set(dataset_details[query_id1]["positives"])
    positives2 = set(dataset_details[query_id2]["positives"])
    find_good_term_subsitutions(query1, semantic_query, end_year1, positives1, output_path=f"{out_path}/generated_semantic_{query_id1}.json")
    find_good_term_subsitutions(query1, chatgpt_query, end_year1, positives1, output_path=f"{out_path}/generated_chatgpt_{query_id1}.json")
    find_good_term_subsitutions(query1, chatgpt_query, end_year1, positives1, output_path=f"{out_path}/generated_autobool_{query_id1}.json")
    find_good_term_subsitutions(query2, manual_query, end_year2, positives2, output_path=f"{out_path}/generated_manual_{query_id2}.json")
    find_good_term_subsitutions(query2, objective_query, end_year2, positives2, output_path=f"{out_path}/generated_objective_{query_id2}.json")
    

    
    
    