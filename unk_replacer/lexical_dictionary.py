import heapq
from collections import defaultdict
from collections import namedtuple
import logging

logger = logging.getLogger(__name__)
Translation = namedtuple('Translation', 'word prob')


class LexicalDictionary:
    def __init__(self, topn=1, vocab=None):
        self.topn = topn
        self.entries = defaultdict(list)
        self.finalized = False
        self.vocab = vocab

    def filter_vocab(self, dic):
        assert self.vocab is not None

        filtered_vocab = []
        for word in dic:
            if len(self.get_translations(word, only_in_vocab=True)) > 0:
                filtered_vocab.append(word)

        return filtered_vocab

    def set_vocab(self, vocab):
        # assert isinstance(vocab, dict), "vocab must be an instance of subclass of dict"
        if isinstance(vocab, list):
            vocab = set(vocab)

        self.vocab = vocab

    def get_translations(self, cond, only_in_vocab=True, topn=None, prob_threshold=0.0):
        """
        Returns topn (word, prob) pairs whose P(word|cond) are the highest.
        If there are less than topn number of pairs, then this method returns all of them
        """

        assert self.finalized, "You have to call self.finalized() to call this method"
        assert (not only_in_vocab) or self.vocab is not None, "You have to set vocab using self.set_vocab() to call this method"

        topn_entries = []

        if topn is None:
            topn = self.topn

        for prob, word in self.entries[cond]:
            if topn is not None and len(topn_entries) >= topn:
                break

            if prob < prob_threshold:
                break

            if only_in_vocab:
                if word in self.vocab:
                    topn_entries.append(Translation(word, prob))
            else:
                topn_entries.append(Translation(word, prob))

        assert topn is None or len(topn_entries) <= topn

        return topn_entries

    def get_prob(self, cond, word):
        # TODO: this is awful code and can be optimized
        for prob, word2 in self.entries[cond]:
            if word == word2:
                return prob

        return 0.0

    def add_entry(self, word, cond, prob):
        assert not self.finalized, "You have already finalized the dictionary"
        assert self.topn is None or len(self.entries[cond]) <= self.topn

        if self.topn is not None and len(self.entries[cond]) == self.topn:
            heapq.heappushpop(self.entries[cond], (prob, word))
        else:
            heapq.heappush(self.entries[cond], (prob, word))

    def finalize(self):
        new_dict = defaultdict(list)
        for cond, heap in self.entries.items():
            while len(heap) > 0:
                new_dict[cond].append(heapq.heappop(heap))

            new_dict[cond] = sorted(new_dict[cond], reverse=True)

        self.entries = new_dict
        self.finalized = True

    @classmethod
    def read_lex_table(cls, filename, topn=1):
        lex_dict = cls(topn=topn)
        with open(filename, "r") as f:
            for line in f:
                word, cond, prob = line.rstrip().split(" ")
                if word == "NULL" or cond == "NULL":
                    continue

                lex_dict.add_entry(word, cond, float(prob))

            lex_dict.finalize()
            return lex_dict
