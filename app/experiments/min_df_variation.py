import json
import os
import pickle
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from app.dataset.utils import load_bow
from app.config.config import BOW_PARAMS


# Test function for different min_df / max_df values
def test_df_values(**bow_args):
    min_df_values = [125]
    max_df_values = [0.5]

    bow = load_bow(**bow_args)
    texts = list(bow.values())

    for min_df in min_df_values:
        for max_df in max_df_values:
            vectorizer = CountVectorizer(
                tokenizer=lambda x: x,
                preprocessor=lambda x: x,
                token_pattern=None,
                binary=True,
                min_df=min_df,
                max_df=max_df,
            )
            X = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()

            # Count occurrences of each feature in the corpus
            word_counts = X.sum(axis=0).A1  # sum over rows
            word_counter = Counter(dict(zip(feature_names, word_counts)))

            most_common_word = word_counter.most_common()[:10]
            least_common_words = word_counter.most_common()[:-110:-1]  # 10 least common

            print(f"min_df={min_df}, max_df={max_df}")
            print(f"  Features left: {len(feature_names)}")
            print(f"  Most common word: {most_common_word}")
            print()
            print(f"  Least common words: {least_common_words}")
            print("-" * 40)


# Example usage
if __name__ == "__main__":
    test_df_values(**BOW_PARAMS)
