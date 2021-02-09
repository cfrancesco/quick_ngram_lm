from collections import Counter
import pickle
import re

from nltk.lm.models import WittenBellInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline
import tqdm

_whitespace_re = re.compile(r'\s+')
import pandas as pd

import numpy as np


class Cleaner:
    def __call__(self, text):
        text = text.lower()
        text = re.sub('[^a-zA-Zäöüß ]+', ' ', text)
        text = re.sub(_whitespace_re, ' ', text)
        text = text.strip()
        text = list(text)
        return text

if __name__ == '__main__':
    # read text
    raw = pd.read_excel('/Users/fcardina/forge/TransformerTTS/private/_snippets.xlsx', skiprows=[0], header=None)
    data = raw[[0, 1]].dropna()
    data_de = data[data[0].apply(lambda x: not x.startswith('en'))]
    wiki = np.array(data_de[1])
    # create data and model
    gram_len = 20
    lm = WittenBellInterpolated(gram_len)
    total = len(wiki)
    print('creating text')
    cleaner = Cleaner()
    text = []
    count = 0
    for line in tqdm.tqdm(wiki, total=total):
        line = cleaner(line)
        if len(line) > 10:
            text.append(line)
        count += 1
        if count > total:
            break
    print('creating everygram pipe')
    train, vocab = padded_everygram_pipeline(gram_len, text)
    print('creating ngram counts')
    counter = Counter()
    for sent in tqdm.tqdm(train, total=total):
        ngrams = [g for g in sent]
        counter.update(ngrams)
    print('fitting language model')
    lm.vocab.update(vocab)
    for ngram, count in tqdm.tqdm(counter.most_common(), total=len(counter)):
        ngram_order = len(ngram)
        if ngram_order == 1:
            lm.counts.unigrams[ngram[0]] += count
        else:
            context, word = ngram[:-1], ngram[-1]
            lm.counts[ngram_order][context][word] += count
    with open(f'/Volumes/data/models/nlp/de/nltk_lm_{gram_len}grams_deasvoice4.pkl', 'wb') as g:
        pickle.dump(lm, g)
