import sys
import os
from app.dataset.utils import load_vectors

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "../systematic-review-datasets")))
from csmed.experiments.csmed_cochrane_retrieval import create_retriever

query = "cancer with legs heart attack"
ret_name = "pubmedbert"
ret_conf = {"type": "dense",
            "model": "pritamdeka/S-PubMedBert-MS-MARCO",
            "max_length": 512
            }
total_docs = 433660
retriever = create_retriever(ret_name, ret_conf, collection=list(range(total_docs)))

ranking = retriever.search(
    query=query, cutoff=10_000, return_docs=False
)
print(ranking)

# X, ordered_pmids, feature_names = load_vectors(total_docs, min_df=min_df, max_df=max_df, mesh=mesh)
# print("Num Features:", len(feature_names), "examples:", feature_names[0], feature_names[-1], feature_names[-2])


# model = GreedyORDecisionTree(**model_args)

# result = train_text_classifier(
#     clf=model,
#     X=X[keep_indices],
#     feature_names=feature_names,
#     labels=np.array(labels),
# )
# tree = result["obj"]
# print("num_positive", sum(labels))
# print("num_negative", len(keep_indices)-num_pos)
# print("precision_dt", result["precision"])
# print("recall_dt", result["recall"])
