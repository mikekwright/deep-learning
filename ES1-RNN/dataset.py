import nltk
import json
import re
import sys

from itertools import islice
from collections import Counter
from nltk import ngrams

nltk.data.path = ['/data/twitter/nltk']


def build_brown_n_grams(n, limit=None):
    words = nltk.corpus.brown.words()
    filtered_words = words if not limit else islice(words, limit)
    grams = ngrams(filtered_words, n)
    return Counter(grams)


def is_valid(ngram_tuple):
    result = True
    for word in ngram_tuple:
        if re.search('[^a-zA-Z0-9]', word):
            result = False
            break
    return result


def build_entry(words, join_character='', alter_func=lambda x: x):
    label = []
    hash_tag = []
    for word in words:
        label.extend([0] * len(word))
        label[-1] = 1
        if len(join_character) > 0:
            label.extend([0] * len(join_character))
            label[-1] = 1

        hash_tag.append(alter_func(word))

    label = label[:len(label)-len(join_character)]
    label[-1] = 0
    return join_character.join(hash_tag), tuple(label)


def build_the_phrases(words):
    ALL_UPPER = lambda x: x.upper()
    ALL_LOWER = lambda x: x.lower()
    CAPITALIZE = lambda x: x.capitalize()

    return { build_entry(word, join_character=char_set, alter_func=operation)
                for word in words
                for char_set in ['', '_', "'s"]
                for operation in [ALL_LOWER, ALL_UPPER, CAPITALIZE] }


def build_entries(freqs):
    ''' generates list of tuples: (actual_phrase, computed_phrase) '''
    for item in freqs.most_common():
        ngram_tuple, count = item
        if is_valid(ngram_tuple):
            yield from build_the_phrases(ngram_tuple)


def build_all_phrases(stream=sys.stdout, min_n=1, max_n=5, limit=100):
    for n in range(min_n, max_n + 1):
        freqs = build_brown_n_grams(n, limit=limit)
        entries = build_entries(freqs)
        for hash_tag, label in entries:
            output = {'hash_tag': hash_tag, 'label': label}
            print(json.dumps(output, ensure_ascii=False), file=stream)

def build_and_save_phrases(output_fpath, min_n=1, max_n=5, limit=100):
    with open(output_fpath, 'w', encoding='utf8') as f:
        build_all_phrases(stream=f, min_n=min_n, max_n=max_n, limit=limit)


# -- Generate test set with
# build_all_phrases('/data/twitter/testset.text', 1, 5, limit=100000)
