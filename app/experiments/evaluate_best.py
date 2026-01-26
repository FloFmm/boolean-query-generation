from app.experiments.evaluate_rf import evalaute_rf


if __name__ == "__main__":
    best_params = []
    run_name = "best"
    for best_p in best_params:
        for query_id in query_ids:
            positives = get_positives(query_id=query_id, dataset=dataset)
            qg_results = evalaute_rf(
                run_name=run_name,
                query_id=query_id,
                X=X,
                positives=positives[query_id],
                feature_names=feature_names,
                sorted_ids=sorted_ids[query_id],
                ordered_pmids=ordered_pmids,
                rf_params=best_p["rf_params"],
                qg_params=best_p["qg_params"],
                term_expansions=term_expansions
            )