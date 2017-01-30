import logging
from os import path
import numpy as np
from gensim import matutils
from gensim.models.word2vec import Word2Vec as GensimWord2Vec
from typing import Iterable, Optional, List
from collections import namedtuple

logger = logging.getLogger(__name__)

MostSimilar = namedtuple('MostSimilar', 'word similarity')


class SentenceGenerator(object):
    def __init__(self, filename):
        self.fname = filename
 
    def __iter__(self):
        for line in open(self.fname, "r"):
            yield line.rstrip().split(" ")


class Word2Vec:

    def __init__(self, model_path: Optional[str]=None, topn: int=1, lowercase_beforehand: bool=False) -> None:
        self.model_path = model_path
        self.topn = topn
        if lowercase_beforehand:
            logger.info("Lowercaser is set")

        self.lowercase_beforehand = lowercase_beforehand
        self.vocab = []  # type: Iterable[str]
        if model_path is None:
            self.model = None
        else:
            logger.info("Loading %s" % model_path)
            self.model = GensimWord2Vec.load(model_path)
            logger.info("Finish loading word embedding model")

    @staticmethod
    def train(model_name: str, training_data: str, size: int=100, window: int=5, negative: int=5, min_count: int=0, workers: int=4) -> None:
        if path.isfile(model_name):
            input("%s already exists and will be overwritten. Press Enter to proceed." % model_name)

        logger.info("Training the model")
        sentences = SentenceGenerator(training_data)
        model = GensimWord2Vec(sentences, size=size, window=window, negative=negative, min_count=min_count, workers=workers)
        logger.info("Saving the model")
        model.init_sims(replace=True) # trim unneeded model memory = use (much) less RAM. 
        # model.save_word2vec_format(model_name, binary=True)
        model.save(model_name)
        logger.info("Trained model was saved to %s." % model_name)

    def most_similar_word(self, word):
        assert self.model is not None, "You have to load a model"
        assert self.vocab is not None, "You have to set vocab"
        assert self.topn is not None
        topn = self.topn

        word = self.lowercase(word)
        try:
            word_vec = self.model.syn0norm[self.model.vocab[word].index]
            dists = np.dot(self.syn0norm_in_vocab, word_vec)
            best_ids = matutils.argsort(dists, topn=topn+1, reverse=True)
            result = [MostSimilar(self.vocab[best_id], float(dists[best_id])) for best_id in best_ids if self.vocab[best_id] != word]
            assert all(vocab in self.vocab for vocab, dist in result)
            if word in self.vocab:
                return [MostSimilar(word, 1.0)] + result[:topn]

            return result[:topn]

        except KeyError:
            logger.info("%s not found in word2vec model" % word)
            return []

    def set_vocab(self, vocab: List[str], topn: int=10000):
        assert self.model is not None
        self.vocab = vocab
        self.model.init_sims()
        vocab_in_word2vec = [word for word in vocab if word in self.model.vocab]
        indices = list(map(lambda word: self.model.vocab[self.lowercase(word)].index, vocab_in_word2vec[:topn]))
        self.syn0norm_in_vocab = self.model.syn0norm[indices]

    def lowercase(self, word: str) -> str:
        if self.lowercase_beforehand:
            return word.lower()
        else:
            return word

    def similarity(self, w1, w2):
        w1 = self.lowercase(w1)
        w2 = self.lowercase(w2)
        try:
            return self.model.similarity(w1, w2)
        except KeyError as e:
            logger.info("%s not in word2vec index" % str(e)) 
            return 0.0
