import os
import random

CUSTOM_HF_PATH = "../systematic-review-datasets/data/huggingface"
os.environ["HF_HOME"] = CUSTOM_HF_PATH  # has to be up here
from collections import defaultdict
from app.dataset.utils import review_id_to_dataset, get_positives, get_dataset_details
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../..", "../systematic-review-datasets"
        )
    )
)
from csmed.experiments.csmed_cochrane_retrieval import load_dataset, build_global_corpus

# from csmed.csmed.csmed_cochrane import CSMED_COCHRANE_REVIEWS
from csmed.csmed_cochrane.prepare_dataset import prepare_dataset


def read_qrels_first_column_set(file_path):
    """
    Reads a qrels file and returns a set of the first column values.

    :param file_path: Path to the qrels file
    :return: Set of values from the first column
    """
    first_column_set = set()

    with open(file_path, "r") as f:
        for line in f:
            if line.strip():  # ignore empty lines
                first_col = line.split()[0]
                first_column_set.add(first_col)

    return first_column_set


def get_filenames_set(path):
    """
    Returns a set of file names (without extensions) in the given directory.

    :param path: Directory path as a string
    :return: Set of file names without extensions
    """
    filenames = []

    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isfile(full_path):
            name_without_ext = os.path.splitext(entry)[0]
            filenames.append(name_without_ext)

    return filenames


def get_tar_review_ids(year=2018):
    if year == 2017:
        tar2017_train = read_qrels_first_column_set(
            "/data/horse/ws/flml293c-master-thesis/tar/2017-TAR/training/qrels/train.combined.qrels"
        )
        tar2017_test = read_qrels_first_column_set(
            "/data/horse/ws/flml293c-master-thesis/tar/2017-TAR/testing/qrels/test.combined.qrels"
        )
        print("Overlap:", len(tar2017_test & tar2017_train))
        print("Number:", len(tar2017_test | tar2017_train))
        print("tar2017_train", sorted(tar2017_train))
        print("tar2017_test", sorted(tar2017_test))
    if year == 2018:
        tar2018_train = set(
            get_filenames_set(
                "/data/horse/ws/flml293c-master-thesis/tar/2018-TAR/Task1/Training/protocols"
            )
        )
        tar2018_train |= set(
            get_filenames_set(
                "/data/horse/ws/flml293c-master-thesis/tar/2018-TAR/Task1/Training/protocols_revised_03092018"
            )
        )
        tar2018_test = set(
            get_filenames_set(
                "/data/horse/ws/flml293c-master-thesis/tar/2018-TAR/Task1/Testing/protocols"
            )
        )
        tar2018_test |= set(
            get_filenames_set(
                "/data/horse/ws/flml293c-master-thesis/tar/2018-TAR/Task1/Testing/protocols_revised_20180708"
            )
        )
        print("Overlap:", len(tar2018_test & tar2018_train))
        print("Number:", len(tar2018_test | tar2018_train))
        print("tar2018_train", sorted(tar2018_train))
        print("tar2018_test", sorted(tar2018_test))


def compute_dataset_statistics():
    dataset = load_dataset()
    gc = build_global_corpus(dataset)
    # print(len(gc))
    global_doc_ids = {d["id"] for d in gc}

    group_stats = defaultdict(
        lambda: {
            "ratio_sum": 0.0,
            "pos_sum": 0,
            "neg_sum": 0,
            "n_reviews": 0,
            "empty_abstract_in_pos": 0,
            "reviews_ge_50_positives": 0,
        }
    )
    missing_relevant_docs = defaultdict(list)
    count_dict = defaultdict(int)
    doc_ids = set()
    review_names = set()

    # r = defaultdict(set)
    for split, reviews in dataset.items():
        print("split", split)
        for review_name, review_data in reviews.items():
            # review_data has keys []'review_name', 'dataset_details', 'data']
            # review_data["dataset_details"] has keys ['title', 'abstract', 'review_type', 'doi', 'review_id', 'criteria', 'search_strategy']
            dataset_name, _, year = review_id_to_dataset(review_name)
            count_dict[dataset_name] += 1

            pos = set()
            neg = set()
            # new_pos = get_positives(dataset=dataset, review_id=review_name)
            for split_name, docs in review_data["data"].items():
                for doc in docs:
                    # doc has keys [review_id pmid, title, abstract, label, mesh_terms]
                    label = int(doc["label"])
                    doc_id = str(doc.get("pmid"))
                    

                    if label == 1:
                        pos.add(doc_id)
                        assert doc_id in global_doc_ids, (
                            f"Missing relevant doc {doc_id} in {dataset_name}"
                        )
                        if len(doc["abstract"]) < 20:
                            group_stats[dataset_name]["empty_abstract_in_pos"] += 1
                            if not (doc["abstract"] == "nan" or doc["abstract"] == "?"):
                                print(doc["abstract"])
                                assert False
                    else:
                        neg.add(doc_id)
            print(len(pos), len(neg))
            neg = len(neg-pos)
            pos = len(pos)
            
            total = pos + neg
            # print(review_name, len(new_pos), pos)
            review_ratio = pos / total

            stats = group_stats[dataset_name]
            stats["ratio_sum"] += review_ratio
            stats["pos_sum"] += pos
            stats["neg_sum"] += neg
            stats["n_reviews"] += 1
            if pos >= 50:
                stats["reviews_ge_50_positives"] += 1

    count_dict["tar2019sum"] = (
        count_dict["tar2019"] + count_dict["tar2018"] + count_dict["tar2017"]
    )
    print(count_dict)
    from app.config.config import CSMED_COCHRANE_REVIEWS

    expected = {
        name: len(set(reviews)) for name, reviews in CSMED_COCHRANE_REVIEWS.items()
    }
    print("expected:", expected)
    print()

    print("\n=== Dataset-level macro statistics ===")

    total_stats = {
        "ratio_sum": 0.0,
        "pos_sum": 0,
        "neg_sum": 0,
        "n_reviews": 0,
        "empty_abstract_in_pos": 0,
        "reviews_ge_50_positives": 0,
    }

    for dataset_name, stats in sorted(group_stats.items()):
        n = stats["n_reviews"]

        avg_ratio = stats["ratio_sum"] / n
        avg_pos = stats["pos_sum"] / n
        avg_neg = stats["neg_sum"] / n
        empty_abstract_in_pos = stats["empty_abstract_in_pos"]
        reviews_ge_50_positives = stats["reviews_ge_50_positives"]

        print(
            f"{dataset_name}: "
            f"avg_ratio={avg_ratio:.3f}, "
            f"avg_pos={avg_pos:.1f}, "
            f"avg_neg={avg_neg:.1f}, "
            f"empty_abstract_in_pos={empty_abstract_in_pos}, "
            f"n_reviews={n}",
            f"reviews_ge_50_positives={reviews_ge_50_positives}",
        )

        for key in total_stats:
            total_stats[key] += stats[key]

    n_total = total_stats["n_reviews"]
    total_pos = total_stats["pos_sum"]
    total_neg = total_stats["neg_sum"]
    print(
        f"TOTAL: "
        f"avg_ratio={total_stats['ratio_sum'] / n_total:.3f}, "
        f"avg_pos={total_pos / n_total:.1f}, "
        f"avg_neg={total_neg / n_total:.1f}, "
        f"empty_abstract_in_pos={total_stats['empty_abstract_in_pos']}, "
        f"n_reviews={n_total}",
        f"reviews_ge_50_positives={total_stats['reviews_ge_50_positives']}",
    )


def compute_train_review_ids(
    total_samples=25,
    min_positives=25,
    seed=42,
):
    random.seed(seed)

    # dataset = load_dataset()
    eligible = defaultdict(list)
    dataset_details = get_dataset_details()

    # 1️⃣ collect eligible reviews per dataset
    # for split, reviews in dataset.items():
    #     for review_name, review_data in reviews.items():
    for review_name, review_data in dataset_details.items():
        dataset_name, _, _ = review_id_to_dataset(review_name)

        # pos = len(get_positives(review_id=review_name, dataset=dataset))
        pos = len(review_data["positives"])

        if pos >= min_positives:
            eligible[dataset_name].append(review_name)

    # 2️⃣ compute totals
    counts = {k: len(v) for k, v in eligible.items()}
    total_eligible = sum(counts.values())

    assert total_eligible >= total_samples, "Not enough eligible reviews"

    # 3️⃣ proportional allocation
    allocation = {
        k: int(round(total_samples * c / total_eligible)) for k, c in counts.items()
    }

    # 4️⃣ fix rounding
    diff = total_samples - sum(allocation.values())

    datasets_sorted = sorted(
        counts.keys(),
        key=lambda k: counts[k],
        reverse=True,
    )

    i = 0
    while diff != 0:
        d = datasets_sorted[i % len(datasets_sorted)]
        if diff > 0:
            allocation[d] += 1
            diff -= 1
        elif allocation[d] > 0:
            allocation[d] -= 1
            diff += 1
        i += 1

    # 5️⃣ sample and PRINT
    sampled = {}

    print("\n=== Selected trains reviews (stratified) ===")
    print(f"Total samples: {total_samples}\n")

    for dataset_name in sorted(allocation.keys()):
        k = allocation[dataset_name]
        selected = random.sample(
            eligible[dataset_name], min(k, len(eligible[dataset_name]))
        )
        sampled[dataset_name] = selected

        print(f"{dataset_name}: {len(selected)} / {counts[dataset_name]} eligible")
        for r in selected:
            # count positives
            pos_count = len(dataset_details[r]["positives"])
            print(f"  - {r} (positives: {pos_count})")
        print()
    print(f"Samples:", sampled)
    return sampled


if __name__ == "__main__":
    compute_dataset_statistics()
    # compute_train_review_ids(total_samples=25, min_positives=50)
