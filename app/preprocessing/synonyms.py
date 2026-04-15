from pathlib import Path

NLTK_DATA_PATH = Path("../systematic-review-datasets/data/nltk_data")
import nltk

# nltk.download("wordnet", download_dir="[Path to your workspace]/systematic-review-datasets/data/nltk_data")
# nltk.download("omw-1.4", download_dir="[Path to your workspace]/systematic-review-datasets/data/nltk_data")
# nltk.download("punkt", download_dir="[Path to your workspace]/systematic-review-datasets/data/nltk_data")
# nltk.download("averaged_perceptron_tagger", download_dir="[Path to your workspace]/systematic-review-datasets/data/nltk_data")
# Tell NLTK to look here first
nltk.data.path.insert(0, str(NLTK_DATA_PATH))
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from app.preprocessing.text_preprocessing import lemmatize_with_synonyms
from collections import defaultdict
import heapq

lemmatizer = WordNetLemmatizer()


def word_net_get_synonyms(word):
    """Return all semantic synonyms of the word."""
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms


def word_net_get_related(word):
    """Return morphological forms of the base word using WordNet."""
    forms = set([word])
    for lemma in wn.lemmas(word):
        forms.add(lemma.name())
        for related in lemma.derivationally_related_forms():
            forms.add(related.name())
    return forms


def transitive_closure(word):
    """Compute transitive closure of related words starting from `word`."""
    closure = set()
    stack = [word]

    while stack:
        w = stack.pop()
        if w not in closure:
            closure.add(w)
            related = word_net_get_related(w)

            stack.extend(related - closure)  # only add unseen words

    return closure


def build_dominating_map(words, related_fn):
    """
    Builds a map from each word to the "dominating" word
    with the largest related words list in which it occurs.

    Args:
        words (list of str): List of words to analyze.
        related_fn (callable): Function(word) -> list of related words.

    Returns:
        dict: Mapping from each word to the dominating word.
    """
    # Step 1: Build word -> related words map
    print("step 1")
    word_map = {}
    for w in words:
        word_map[w] = sorted(related_fn(w))

    # Step 2: Build reverse map: word -> all words whose related list contains it
    print("step 2")
    reverse_map = {}
    for word, related_words in word_map.items():
        for rw in related_words:
            reverse_map.setdefault(rw, []).append(word)

    # Step 3: For each word, find the dominating word
    print("step 3")
    dominating_map = {}
    dominating_map_reversed = defaultdict(set)
    for word, candidates in reverse_map.items():
        max_word = max(candidates, key=lambda w: len(word_map[w]))
        dominating_map[word] = max_word
        dominating_map_reversed[max_word].add(word)

    return dominating_map, {k: list(v) for k, v in dominating_map_reversed.items()}


# def nltk_pos_to_wordnet(pos_tag):
#     if pos_tag.startswith('N'):
#         return wn.NOUN
#     if pos_tag.startswith('V'):
#         return wn.VERB
#     if pos_tag.startswith('J'):
#         return wn.ADJ
#     if pos_tag.startswith('R'):
#         return wn.ADV
#     return None

# def lemmatize_text(text):
#     tokens = word_tokenize(text)
#     tagged = pos_tag(tokens)
#     lemmatized = {}
#     for word, tag in tagged:
#         wn_pos = nltk_pos_to_wordnet(tag)
#         print(wn_pos)
#         base, forms = word_net_get_related(word, wn_pos)
#         lemmatized[base] = forms
#     return lemmatized
import json


def process_synonym_file(file_path, out_file, related_fn):
    """
    Reads a JSON synonym dictionary, computes dominating map on keys,
    and overwrites the file with the result.
    """
    file_path = Path(file_path)
    out_file = Path(out_file)

    # Read the JSON
    with file_path.open("r", encoding="utf-8") as f:
        synonym_dict = json.load(f)

    # Get keys
    words = list(synonym_dict.keys())

    # Build dominating map
    dom_map, reverse_map = build_dominating_map(words, related_fn)
    reverse_map = dict(sorted(reverse_map.items()))
    # Save back
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(reverse_map, f, ensure_ascii=False, indent=2)

    print(f"Dominating map saved to {out_file}")


# Example usage
# print(word_net_get_related("maintainable"))
# print()
# print(transitive_closure("maintainable"))
# text = "correct correctability correctable corrected correctible correcting correction corrections corrective correctives correctness corrects"
# text = """
# Abstract

# Psychological studies have demonstrated that expectations can have substantial effects on choice behavior, although the role of expectations on social decision making in particular has been relatively unexplored. To broaden our knowledge, we examined the role of expectations on decision making when interacting with new game partners and then also in a subsequent interaction with the same partners. To perform this, 38 participants played an Ultimatum Game (UG) in the role of responders and were primed to expect to play with two different groups of proposers, either those that were relatively fair (a tendency to propose an equal split-the high expectation condition) or unfair (with a history of offering unequal splits-the low expectation condition). After playing these 40 UG rounds, they then played 40 Dictator Games (DG) as allocator with the same set of partners. The results showed that expectations affect UG decisions, with a greater proportion of unfair offers rejected from the high as compared to the low expectation group, suggesting that players utilize specific expectations of social interaction as a behavioral reference point. Importantly, this was evident within subjects. Interestingly, we also demonstrated that these expectation effects carried over to the subsequent DG. Participants allocated more money to the recipients of the high expectation group as well to those who made equal offers and, in particular, when the latter were expected to behave unfairly, suggesting that people tend to forgive negative violations and appreciate and reward positive violations. Therefore, both the expectations of others' behavior and their violations play an important role in subsequent allocation decisions. Together, these two studies extend our knowledge of the role of expectations in social decision making.

# Keywords: Dictator Game; Ultimatum Game; expectations; social decision-making.
# """
# for t in text.split(" "):
#     print(transitive_closure(t))
# config = {
#     "lower_case": True,
#     "mesh_ancestors": True,
#     "rm_numbers": True,
#     "rm_punct": True,
# }
# result = lemmatize_with_synonyms(text, conf=config)
# lemmas = result.keys()
# "../systematic-review-datasets/data/bag_of_words/synonym_map,d=433660_old.jsonl"
# result = build_dominating_map(lemmas, transitive_closure)
# print(result)

# process_synonym_file(
#     "../systematic-review-datasets/data/bag_of_words/synonym_map,d=433660_old.json",
#     "../systematic-review-datasets/data/bag_of_words/synonym_result.json",
#     transitive_closure
# )

# # Step 1: invert the mapping to know which words each alternative can cover
# cover_map = defaultdict(set)
# for word, alts in word_map.items():
#     for alt in alts:
#         cover_map[alt].add(word)

# # Step 2: Greedy selection using a heap (max-heap by coverage)
# covered = set()
# result = {}

# # Make a heap of (-coverage_count, word)
# heap = [(-len(words), word) for word, words in cover_map.items()]
# heapq.heapify(heap)

# while len(covered) < len(word_map):
#     while heap:
#         neg_count, candidate = heapq.heappop(heap)
#         uncovered = cover_map[candidate] - covered
#         if uncovered:
#             # This candidate covers new words
#             result[candidate] = list(uncovered)
#             covered.update(uncovered)
#             break

# print(result)

# print()
# # exit(0)
# word = "correcting"
# print("Lemmas:", sorted(word_net_get_related(word)))
# print("Synonyms:", sorted(word_net_get_synonyms(word)))
