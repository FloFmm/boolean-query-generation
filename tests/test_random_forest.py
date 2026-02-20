import numpy as np
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score
from app.tree_learning.random_forest import RandomForest
from app.tree_learning.disjunctive_dt import generate_texts_from_boolean
from app.tree_learning.query_generation import (
    extract_and_vectorize_rules,
    select_rules_via_ga,
    rules_to_pubmed_query,
)

if __name__ == "__main__":

    def f(d):
        return (
            not (d["cats"] or d["dogs"] or d["mice[mh]"])
            and (d["house"] or d["wohnung"])
            and (d["bowl"] or d["box"])
        )

    variables = ["cats", "dogs", "mice[mh]", "house", "wohnung", "bowl", "box"]
    texts, labels = generate_texts_from_boolean(
        func=f,
        variables=variables,
        error=0.0,
        completeness=1.0,
        seed=42,
        doc_count=5_000,
        word_pool_size=1_000,
        average_doc_length=50,
    )

    # --- Calculate actual statistics from the result ---
    actual_doc_count = len(texts)
    all_words = [word for text in texts for word in text]
    unique_words = set(all_words)
    actual_word_pool_size = len(unique_words)
    avg_doc_length = sum(len(text) for text in texts) / len(texts)

    # --- Print for verification ---
    print("Actual doc_count:", actual_doc_count)
    print("Actual word_pool_size:", actual_word_pool_size)
    print("Actual average_doc_length:", round(avg_doc_length, 2))

    # vectorizer = CountVectorizer(binary=True)
    # X = vectorizer.fit_transform(texts)
    vectorizer = CountVectorizer(
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
        token_pattern=None,
        binary=True,
        # stop_words="english",
        # min_df=1,
        # max_df=0.6
    )
    X = vectorizer.fit_transform(texts)

    forest = RandomForest(
        n_estimators=20,
        max_depth=4,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0005,
        max_features="sqrt",
        min_impurity_decrease_range_start=0.01,
        min_impurity_decrease_range_end=0.01,
        bootstrap=False,
        n_jobs=None,
        random_state=None,
        verbose=True,
        class_weight="balanced",
        ccp_alpha=0.0,
        max_samples=None,
        top_k_or_candidates=500,
    )
    start_time = time.time()
    forest.fit(X, np.array(labels), feature_names=vectorizer.get_feature_names_out())
    end_time = time.time()

    print()
    for tree in forest.estimators_:
        print()
        print(tree.pretty_print(verbose=True, prune=True))

    print()
    print(end_time - start_time)
    print()
    print("FOREST:", forest.pubmed_query()[0])
    print()
    print("FOREST (optimized):", forest.pubmed_query(X=X, labels=labels)[0])
    print()
    # vec_result = extract_and_vectorize_rules(forest=forest, X=X)
    # rules = vec_result["rules"]
    # kept_variables = vec_result["kept_variables"]
    # coverage = vec_result["coverage"]
    # print(coverage)

    # result = select_rules_via_ga(
    #     coverage = coverage,
    #     y=np.array(labels)
    # )
    # print("============RESULT RULES===========")
    # for i in result["selected_rule_indices"]:
    #     print(rules_to_pubmed_query([rules[i]])[0])

