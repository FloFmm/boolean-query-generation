import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score
import numpy as np
from app.tree_learning.disjunctive_dt import generate_texts_from_boolean, GreedyORDecisionTree
from app.pubmed.utils import pubmed_query_to_lambda, remove_tags

def run_tree_test(pubmed_query, 
                  text_gen_params={"error": 0.0,
                    "completeness": 1.0,
                    "seed": 42,
                    "doc_count": 500_000,
                    "word_pool_size": 5,
                    "average_doc_length": 5}, 
                  tree_params={"max_depth":4,
                    "min_impurity_decrease_range": [0.01, 0.03],
                    "top_k_or_candidates": 500,
                    "verbose": True,
                    "min_samples_split": 1,
                    "class_weight": "balanced"}):

    f, variables = pubmed_query_to_lambda(pubmed_query)

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
        
        true_preds = np.array(
            [f({var: int(var in text) for var in variables}) for text in texts]
        )
        precision = precision_score(true_preds, preds)
        recall = recall_score(true_preds, preds)
        
        
        # Indices where prediction is wrong
        wrong_idx = np.where(preds != true_preds)[0]
        # Print first N errors
        N = 20
        for i in wrong_idx[:N]:
            print("WRONG PREDICTION", "=" * 60)
            print(f"Index: {i}")
            print(f"Text: {texts[i]}")
            print(f"True: {true_preds[i]}")
            print(f"Pred: {preds[i]}")
        
        assert precision == 1.0
        assert recall == 1.0
    
        # evalaute pubmed query
        best_threshold, best_score, final_constraint_score = t._find_optimal_threshold(
            X,
            np.array(labels),
            metric="f3",#"pubmed_f2",
            # constraint=constraint,#"pubmed_count",-1 * 50_000,#num_pos*50,
            # term_expansions=synonym_map
        )
        # Generate PubMed query
        generated_pubmed_query, query_size = t.pubmed_query()
        generated_pubmed_query = remove_tags(generated_pubmed_query)
        generated_f, generated_variables = pubmed_query_to_lambda(generated_pubmed_query)
        
        np.testing.assert_array_equal(
            true_preds,
            np.array(
                [generated_f({var: int(var in text) for var in variables}) for text in texts]
            )
        )
        assert variables == generated_variables
        assert len(pubmed_query.strip(" )(")) >= len(generated_pubmed_query.strip(" )("))
    
def test_basic_formulas():
    formulas = [
        """((cats OR dogs OR mice) NOT (bowl OR box OR house OR wohnung)) OR ((bowl OR box) AND (house OR wohnung) NOT (cats OR dogs OR mice))""",
        """NOT (cats OR dogs OR mice) AND (house OR wohnung) AND (bowl OR box)""",
        """(A AND (B OR C)) OR (NOT A AND C) OR D""",
        """NOT A AND B""",
        """A""",
        """XX OR (YY NOT XX NOT (AA OR BB OR CC OR DD))""",
        """hello AND bye AND (nope OR never)"""
    ]

    # variables = ["cats", "dogs", "mice[mh]", "house", "wohnung", "bowl", "box", "nothing"]
    text_params_list = [{"error": 0.0,
                    "completeness": 1.0,
                    "seed": 42,
                    "doc_count": 100_000,
                    "word_pool_size": 10_000,
                    "average_doc_length": 60}]
    tree_params_list = [{"max_depth":4,
                    "min_impurity_decrease_range": [0.001, 0.001],
                    "top_k_or_candidates": 500,
                    "verbose": True,
                    "min_samples_split": 1,
                    "class_weight": "balanced"}]
    for f in formulas:
        for text_params in text_params_list:
            for tree_params in tree_params_list:
                run_tree_test(f, text_gen_params=text_params, tree_params=tree_params)
    
if __name__ == "__main__":
    test_basic_formulas()
 