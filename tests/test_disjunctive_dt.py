import time
import pytest
import inspect
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score
import numpy as np
from app.tree_learning.disjunctive_dt import (
    generate_texts_from_boolean,
    GreedyORDecisionTree,
)
from app.tree_learning.random_forest import RandomForest
from app.pubmed.utils import pubmed_query_to_lambda, remove_tags


def run_tree_test(
    classifier,
    pubmed_query,
    text_gen_params={
        "error": 0.0,
        "completeness": 1.0,
        "seed": 42,
        "doc_count": 500_000,
        "word_pool_size": 5,
        "average_doc_length": 5,
    },
    qg_params={"min_tree_occ": 0.05, "min_rule_occ": 0.05, "cost_factor": 0.002},
):
    f, variables = pubmed_query_to_lambda(pubmed_query)

    texts, labels = generate_texts_from_boolean(
        func=f, variables=variables, **text_gen_params
    )

    vectorizer = CountVectorizer(
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
        token_pattern=None,
        binary=True,
    )
    X = vectorizer.fit_transform(texts)

    # tree = GreedyORDecisionTree(**tree_params)
    feature_names = vectorizer.get_feature_names_out()
    classifier.fit(
        X, np.array(labels), feature_names=feature_names
    )

    # tree_json = tree.to_json()
    # loaded_tree = GreedyORDecisionTree.from_json(tree_json)
    # preds = classifier.predict(X)

    true_preds = np.array(
        [f({var: int(var in text) for var in variables}) for text in texts]
    )
    # precision = precision_score(true_preds, preds)
    # recall = recall_score(true_preds, preds)

    # # Indices where prediction is wrong
    # wrong_idx = np.where(preds != true_preds)[0]
    # # Print first N errors
    # N = 20
    # for i in wrong_idx[:N]:
    #     print("WRONG PREDICTION", "=" * 60)
    #     print(f"Index: {i}")
    #     print(f"Text: {texts[i]}")
    #     print(f"True: {true_preds[i]}")
    #     print(f"Pred: {preds[i]}")

    # assert precision == 1.0
    # assert recall == 1.0

    # evalaute pubmed query
    # classifier._find_optimal_threshold(
    #     X=X,
    #     y_true=np.array(labels),
    #     metric="f3",#"pubmed_f2",
    #     # constraint=constraint,#"pubmed_count",-1 * 50_000,#num_pos*50,
    #     # term_expansions=synonym_map
    # )
    # Generate PubMed query
    sig = inspect.signature(classifier.pubmed_query)
    if len(sig.parameters) == 1:
        generated_pubmed_query, query_size = classifier.pubmed_query()
    else:
        generated_pubmed_query, query_size = classifier.pubmed_query(
            X=X,
            labels=labels,
            feature_names=feature_names,
            min_tree_occ=qg_params["min_tree_occ"],
            min_rule_occ=qg_params["min_rule_occ"],
            cost_factor=qg_params["cost_factor"],
        )
        
    generated_pubmed_query = remove_tags(generated_pubmed_query)
    print(classifier.estimators_[0].pretty_print(verbose=True))
    print()
    print("CORRECT QUERY\n", pubmed_query)
    print()
    print("GENERATED QUERY\n", generated_pubmed_query)
    print()
    generated_f, generated_variables = pubmed_query_to_lambda(generated_pubmed_query)
    # print(variables)
    # print()
    # print(generated_variables)
    # exit(0)
    np.testing.assert_array_equal(
        true_preds,
        np.array(
            [
                generated_f({var: int(var in text) for var in generated_variables})
                for text in texts
            ]
        ),
    )
    assert variables == generated_variables
    assert len(pubmed_query.strip(" )(")) >= len(generated_pubmed_query.strip(" )("))


TEXT_PARAMS = {
    "error": 0.0,
    "completeness": 1.0,
    "seed": 42,
    "doc_count": 10_000,
    "word_pool_size": 1_000,
    "average_doc_length": 60,
}
TREE_PARAMS = {
    "max_depth": 4,
    "min_impurity_decrease_range": [0.001, 0.001],
    "top_k_or_candidates": 500,
    "verbose": True,
    "min_samples_split": 1,
    "class_weight": "balanced",
}
RF_PARAMS = {
    "n_estimators": 10,
    "max_depth": 4,
    "min_samples_split": 2,
    "min_weight_fraction_leaf": 0.0005,
    "max_features": "sqrt",
    "randomize_max_feature": 1,
    "min_impurity_decrease_range": (0.01, 0.01),
    "randomize_min_impurity_decrease_range": 1,
    "bootstrap": True,
    "n_jobs": None,
    "random_state": None,
    "verbose": False,
    "class_weight": "balanced",
    "ccp_alpha": 0.0,
    "max_samples": None,
    "top_k_or_candidates": 500,
}
QG_PARAMS = {
    "min_tree_occ": 0.05,
    "min_rule_occ": 0.05,
    "cost_factor": 0.002, # 50 ANDs are worth 0.1 F3 score
    
    }

FORMULAS = [
    """((cats OR dogs OR mice) NOT (bowl OR box OR house OR wohnung)) OR ((bowl OR box) AND (house OR wohnung) NOT (cats OR dogs OR mice))""",
    """NOT (cats OR dogs OR mice) AND (house OR wohnung) AND (bowl OR box)""",
    """(A AND (B OR C)) OR (NOT A AND C) OR D""",
    """NOT A AND B""",
    """A""",
    """XX OR (YY NOT XX NOT (AA OR BB OR CC OR DD))""",
    """hello AND bye AND (nope OR never)""",
    """(A AND B) OR (C AND D) OR (E AND F)""",
    """A NOT B""",
]

@pytest.mark.parametrize("formula", FORMULAS, ids=lambda f: f[:40])
def test_basic_formulas_dt(formula):
    tree = GreedyORDecisionTree(**TREE_PARAMS)
    run_tree_test(tree, formula, text_gen_params=TEXT_PARAMS)


@pytest.mark.parametrize("formula", FORMULAS, ids=lambda f: f[:40])
def test_basic_formulas_rf(formula):
    tree = RandomForest(**RF_PARAMS)
    run_tree_test(
        tree, formula, text_gen_params=TEXT_PARAMS, qg_params=QG_PARAMS
    )


if __name__ == "__main__":
    for f in FORMULAS:
        test_basic_formulas_rf(formula=f)
