import pickle

import numpy as np


def generate(gen_len=None, starting_context=None, context_len=2, seed=None, noise=0.):
    sentence = ''
    if starting_context is None:
        starting_context = ('<s>',) * (context_len - 1)
    context = starting_context
    next_letter = None
    if gen_len is None:
        gen_len = np.inf
    np.random.seed(seed)
    while next_letter != '</s>' and len(sentence) < gen_len:
        candidates = dict(lm.counts[context_len][context])
        if '<s>' in candidates:
            candidates.pop('<s>')
        vals = np.array(list(candidates.values()))
        probs = vals / np.linalg.norm(vals, 1)
        if noise > 0.:
            probs += np.random.normal(size=len(probs)) * noise
            probs += abs(min(probs))
            probs = probs / np.linalg.norm(probs, 1)
        next_letter = np.random.choice(list(candidates.keys()), p=probs)
        sentence += next_letter
        context = context[1:] + (next_letter,)
    return sentence[:-4]


if __name__ == '__main__':
    # lm = pickle.load(open('/Volumes/data/models/nlp/de/nltk_lm_dewiki.pkl', 'rb'))
    # lm = pickle.load(open('/Volumes/data/models/nlp/de/nltk_lm_10grams_deasvoice4.pkl', 'rb'))
    lm = pickle.load(open('/Volumes/data/models/nlp/de/nltk_lm_20grams_deasvoice4.pkl', 'rb'))
    generate(context_len=20, noise=0.05)
