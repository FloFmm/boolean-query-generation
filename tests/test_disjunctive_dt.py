import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score
import numpy as np
from app.tree_learning.disjunctive_dt import generate_texts_from_boolean, GreedyORDecisionTree

def run_tree_test(f, 
                  variables, 
                  text_gen_params={"error": 0.0,
                    "completeness": 1.0,
                    "seed": 42,
                    "doc_count": 500_000,
                    "word_pool_size": 50_000,
                    "average_doc_length": 60}, 
                  tree_params={"max_depth":4,
                    "min_impurity_decrease_range": [0.01, 0.03],
                    "top_k_or_candidates": 500,
                    "verbose": True,
                    "min_samples_split": 1,
                    "class_weight": "balanced"}):

    texts, labels = generate_texts_from_boolean(
        func=f,
        variables=variables,
        **text_gen_params
    )

    vectorizer = CountVectorizer(
        tokenizer=lambda x: x, 
        preprocessor=lambda x: x,
        token_pattern=None,
        binary=True,
    )
    X = vectorizer.fit_transform(texts)

    tree = GreedyORDecisionTree(**tree_params)
    tree.fit(X, np.array(labels), feature_names=vectorizer.get_feature_names_out())

    tree_json = tree.to_json()
    loaded_tree = GreedyORDecisionTree.from_json(tree_json)
    for t in [tree, loaded_tree]:
        preds = t.predict(X)
        loaded_tree_preds = loaded_tree.predict(X)
        
        true_preds = np.array(
            [f({var: int(var in text) for var in variables}) for text in texts]
        )
        precision = precision_score(true_preds, preds)
        recall = recall_score(true_preds, preds)
        loaded_tree_precision = precision_score(true_preds, loaded_tree_preds)
        loaded_tree_recall = recall_score(true_preds, loaded_tree_preds)
        assert precision == 1.0
        assert recall == 1.0
        assert loaded_tree_precision == 1.0
        assert loaded_tree_recall == 1.0
    
        # evalaute pubmed query
        best_threshold, best_score, final_constraint_score = t._find_optimal_threshold(
            X,
            np.array(labels),
            metric="f3",#"pubmed_f2",
            # constraint=constraint,#"pubmed_count",-1 * 50_000,#num_pos*50,
            # term_expansions=synonym_map
        )
        # Generate PubMed query
        pubmed_query_str, query_size = t.pubmed_query()
        print(pubmed_query_str)
    
    print("passed")
    
if __name__ == "__main__":
    formulas = [
        lambda d :
            not (d["cats"] or d["dogs"] or d["mice[mh]"]) and
            (d["house"] or d["wohnung"])
            and (d["bowl"] or d["box"]),  
        lambda d :
            not (d["cats"] or d["dogs"] or d["mice[mh]"]) and
            (d["house"] or d["wohnung"])
            and (d["bowl"] or d["box"]) or 
            (d["cats"] or d["dogs"] or d["mice[mh]"]) and
            not (d["house"] or d["wohnung"])
            and not (d["bowl"] or d["box"]),  
    ]

    variables = ["cats", "dogs", "mice[mh]", "house", "wohnung", "bowl", "box", "nothing"]
    text_params_list = [{"error": 0.0,
                    "completeness": 1.0,
                    "seed": 42,
                    "doc_count": 500_000,
                    "word_pool_size": 50_000,
                    "average_doc_length": 60}]
    tree_params_list = [{"max_depth":4,
                    "min_impurity_decrease_range": [0.01, 0.03],
                    "top_k_or_candidates": 500,
                    "verbose": True,
                    "min_samples_split": 1,
                    "class_weight": "balanced"}]
    for f in formulas:
        for text_params in text_params_list:
            for tree_params in tree_params_list:
                run_tree_test(f, variables)#, text_gen_params=text_params, tree_params=tree_params)
 