import argparse
import json

from collections import defaultdict
import math
from itertools import zip_longest


class BleuComputer(object):
    def __init__(self):
        self.ngrams_corrects = {1: 0, 2: 0, 3: 0, 4: 0}
        self.ngrams_total = {1: 0, 2: 0, 3: 0, 4: 0}
        self.total_length = 0
        self.ref_length = 0

    def copy(self):
        res = BleuComputer()
        res.ngrams_corrects = self.ngrams_corrects.copy()
        res.ngrams_total = self.ngrams_total.copy()
        res.total_length = self.total_length
        res.ref_length = self.ref_length
        return res

    def __repr__(self):
        res = []
        res.append("bleu:%f%%   " % (self.bleu() * 100))
        for n in range(1, 5):
            if self.ngrams_total[n] == 0:
                assert self.ngrams_corrects[n] == 0
                ratio_n = 1
            else:
                ratio_n = float(self.ngrams_corrects[n]) / self.ngrams_total[n]
            res.append("%i/%i[%f%%]" % (self.ngrams_corrects[n], self.ngrams_total[n], 100.0 * ratio_n))
        res.append("size of cand/ref: %i/%i[%f]" % (
        self.total_length, self.ref_length, float(self.total_length) / self.ref_length))
        return " ".join(res)

    __str__ = __repr__

    def bleu(self):
        if min(self.ngrams_corrects.values()) <= 0:
            return 0
        assert min(self.ngrams_total.values()) >= 0
        assert min(self.ngrams_total.values()) >= min(self.ngrams_corrects.values())

        log_brevity_penalty = min(0, 1.0 - float(self.ref_length) / self.total_length)
        log_average_precision = 0.25 * (
            sum(math.log(v) for v in self.ngrams_corrects.values()) -
            sum(math.log(v) for v in self.ngrams_total.values())
        )
        res = math.exp(log_brevity_penalty + log_average_precision)
        return res

    def update(self, reference, translation):
        self.ref_length += len(reference)
        self.total_length += len(translation)
        for n in range(1, 5):
            reference_ngrams = defaultdict(int)
            translation_ngrams = defaultdict(int)
            for start in range(0, len(reference) - n + 1):
                ngram = tuple(reference[start: start + n])
                reference_ngrams[ngram] += 1
            for start in range(0, len(translation) - n + 1):
                ngram = tuple(translation[start: start + n])
                translation_ngrams[ngram] += 1
            for ngram, translation_freq in translation_ngrams.iteritems():
                reference_freq = reference_ngrams[ngram]
                self.ngrams_total[n] += translation_freq
                if ngram in reference_ngrams:
                    if reference_freq >= translation_freq:
                        self.ngrams_corrects[n] += translation_freq
                    else:
                        self.ngrams_corrects[n] += reference_freq

    @classmethod
    def get_bc(cls, ref_fn, trans_fn):
        ref_file = open(ref_fn)
        trans_file = open(trans_fn)
        bc = BleuComputer()
        for line_ref, line_trans, log in zip_longest(ref_file, trans_file, replace_log_file):
            r = line_ref.strip().split(" ")
            t = line_trans.strip().split(" ")
            bc.update(r, t, log)

        return bc

    @classmethod
    def get_detailed_bc(cls, ref_fn, trans_fn, replace_log_fn):
        ref_file = open(ref_fn)
        trans_file = open(trans_fn)
        replace_log_file = json.load(open(replace_log_fn))
        bc_changed = BleuComputer()
        bc_not_changed = BleuComputer()
        bc_total = BleuComputer()
        for line_ref, line_trans, log in zip_longest(ref_file, trans_file, replace_log_file):
            r = line_ref.strip().split(" ")
            t = line_trans.strip().split(" ")
            if len(log) > 0:
                bc_changed.update(r, t)
            else:
                bc_not_changed.update(r, t)

            bc_total.update(r, t)

        return {"total": bc_total,
                "changed": bc_changed,
                "not_changed": bc_not_changed}


def main():
    parser = argparse.ArgumentParser(description="Compute BLEU score")
    parser.add_argument("ref")
    parser.add_argument("translations")
    parser.add_argument('--replace-log', type=str, default=None, help="Path to replace log")
    options = parser.parse_args()

    if options.replace_log is None:
        bc = BleuComputer.get_bc(options.ref, options.translations)
        print(bc)
    else:
        bc = BleuComputer.get_detailed_bc(options.ref, options.translations, options.replace_log)
        print("BLEU on the sentences including replaced words")
        print(bc["changed"])
        print("BLEU on the sentences without replaced words")
        print(bc["not_changed"])
        print("BLEU for all sentences")
        print(bc["total"])


if __name__ == "__main__":
    main()
