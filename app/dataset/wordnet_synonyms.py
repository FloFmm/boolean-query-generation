from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def word_net_get_related(word, pos='v'):
    """Return morphological forms of the base word using WordNet."""
    base = lemmatizer.lemmatize(word, pos=pos) #TODO get part of speech tag
    
    forms = set()
    for lemma in wn.lemmas(base):
        forms.add(lemma.name())
        for related in lemma.derivationally_related_forms():
            forms.add(related.name())
    return forms

def word_net_get_synonyms(word):
    """Return all semantic synonyms of the word."""
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms

# Example usage
word = "correcting"
print("Lemmas:", sorted(word_net_get_related(word)))
print("Synonyms:", sorted(word_net_get_synonyms(word)))
