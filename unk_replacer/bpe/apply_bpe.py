# Author: Rico Sennrich

"""Use operations learned with learn_bpe.py to encode a new text.
The text will not be smaller, but use only a fixed vocabulary, with rare words
encoded as variable-length sequences of subword units.

Reference:
Rico Sennrich, Barry Haddow and Alexandra Birch (2015). Neural Machine Translation of Rare Words with Subword Units.
Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
"""

import sys
import argparse
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class BPE(object):

    def __init__(self, vocab, separator='@@', use_separator=False, unk_symbol="<unk>", eow="</w>"):
        
        self.bpe_codes = []
        self.bpe_subwords = set()
        for item in vocab:
            if len(item) == 2:
                self.bpe_codes.append(tuple(item))

            self.bpe_subwords.add(''.join(item))

        # some hacking to deal with duplicates (only consider first instance)
        self.bpe_codes = dict([(code, i) for (i, code) in reversed(list(enumerate(self.bpe_codes)))])

        self.separator = separator
        self.use_separator = use_separator
        self.unk_symbol = unk_symbol
        self.eow = eow
        self.cache = {}

    def _process_unk(self, word):
        if word in self.bpe_subwords:
            return word
        else:
            logger.info("%s is unknown" % word)
            return self.unk_symbol

    def get_vocab(self):
        return self.bpe_subwords

    def segment(self, sentence):
        """segment single sentence (whitespace-tokenized string) with BPE encoding"""

        output = [self.segment_word(word) for word in sentence.strip(" \t\n").split(" ")]
        return ' '.join(output)

    def segment_word(self, word):
        """segment single word(whitespace-tokenized string) with BPE encoding"""
        output = []
        new_word = encode(word, self.bpe_codes, self.cache)
        if self.use_separator:
            for item in new_word[:-1]:
                output.append(self._process_unk(item) + self.separator)

            last_token = self._process_unk(new_word[-1])
            output.append(last_token)
        else:
            for item in new_word:
                output.append(self._process_unk(item))
        
        return ' '.join(output)


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="learn BPE-based word segmentation")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input file (default: standard input).")
    parser.add_argument(
        '--codes', '-c', type=argparse.FileType('r'), metavar='PATH',
        required=True,
        help="File with BPE codes (created by learn_bpe.py).")
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="Output file (default: standard output)")
    parser.add_argument(
        '--separator', '-s', type=str, default='@@', metavar='STR',
        help="Separator between non-final subword units (default: '%(default)s')")
    parser.add_argument(
        '--unk-symbol', '-u', type=str, default='<unk>', metavar='STR',
        help="Unknown symbol (default: '%(default)s')"
    )
    parser.add_argument(
        '--use-separator', '-e', action='store_true', default=False,
        help="Each subword is appended by separator (default: '%(default)s')"
    )
    parser.add_argument(
        '--eow', '-w' , type=str, default='</w>', metavar='STR',
        help="End of word token (default: '%(default)s')"
    )

    return parser


def get_pairs(word):
    """Return set of symbol pairs in a word.

    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def encode(orig, bpe_codes, cache):
    """Encode word based on list of BPE merge operations, which are applied consecutively
    """
    if orig in cache:
        return cache[orig]

    word = tuple(orig) + ('</w>',)
    pairs = get_pairs(word)

    while True:
        bigram = min(pairs, key = lambda pair: bpe_codes.get(pair, float('inf')))
        if bigram not in bpe_codes:
            break
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i = j
            except:
                new_word.extend(word[i:])
                break

            if word[i] == first and i < len(word)-1 and word[i+1] == second:
                new_word.append(first+second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)
        word = new_word
        if len(word) == 1:
            break
        else:
            pairs = get_pairs(word)

    cache[orig] = word
    return word


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    bpe = BPE(args.codes, args.separator, use_separator=args.use_separator, unk_symbol=args.unk_symbol, eow=args.eow)

    for line in args.input:
        args.output.write(bpe.segment(line).strip(" \t\n"))
        args.output.write('\n')
