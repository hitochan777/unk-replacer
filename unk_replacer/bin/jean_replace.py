import argparse
import re
from typing import List
import logging

from unk_replacer.lexical_dictionary import LexicalDictionary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JeanReplacer:

    def __init__(self, lex_e2f: LexicalDictionary, lex_f2e: LexicalDictionary, unk_tag_pattern):
        self.lex_e2f = lex_e2f  # type: LexicalDictionary
        self.lex_f2e = lex_f2e  # type: LexicalDictionary
        self.unk_tag_pattern = unk_tag_pattern
        self.cache = {}

    def get_best_translation(self, orig_word):
        if orig_word in self.cache:
            return self.cache[orig_word]

        candidates = self.lex_f2e.get_translations(orig_word, only_in_vocab=False)
        max_prob = float("-inf")
        best_word = None
        for word, f2e_prob in candidates:
            if self.lex_e2f is None:
                e2f_prob = 1.0
            else:
                e2f_prob = self.lex_e2f.get_prob(cond=word, word=orig_word)

            cur_prob = f2e_prob * e2f_prob
            if cur_prob > max_prob:
                max_prob = cur_prob
                best_word = word

        if best_word is not None:
            self.cache[orig_word] = best_word
        else:
            self.cache[orig_word] = orig_word

        return self.cache[orig_word]

    def replace(self, translation: List[str], orig: List[str]) -> List[str]:
        final_translation = list(translation)  # clone translation
        for e_index, tgt_word in enumerate(translation):
            match = re.search(self.unk_tag_pattern, tgt_word)
            if match is None:
                continue

            f_index = int(match.group("f_index"))
            orig_word = orig[f_index]
            best_word = self.get_best_translation(orig_word)
            final_translation[e_index] = best_word

        return final_translation

    def clear_cache(self):
        self.cache = {}


def main(args=None):
    parser = argparse.ArgumentParser(description='Implementation of [Jean+ 2015]')
    parser.add_argument('--translation', required=True, type=str, help='Path to raw translation')
    parser.add_argument('--input', required=True, type=str, help='Original input')
    parser.add_argument('--f2e', required=True, type=str, help='Path to f2e dictionary')
    parser.add_argument('--e2f', default=None, type=str, help='Path to e2f dictionary')
    parser.add_argument('--unk-tag-pattern', default="#T_UNK_(?P<f_index>\d+)#",
                        help='Regex for UNK symbol. default: %(default)s')

    options = parser.parse_args(args)


    if options.e2f is not None:
        logger.info("Loading e2f lexical dictionary")
        lex_e2f = LexicalDictionary.read_lex_table(options.e2f, topn=None)
    else:
        lex_e2f = None

    logger.info("Loading f2e lexical dictionary")
    lex_f2e = LexicalDictionary.read_lex_table(options.f2e, topn=None)

    replacer = JeanReplacer(lex_e2f=lex_e2f, lex_f2e=lex_f2e, unk_tag_pattern=options.unk_tag_pattern)

    with open(options.translation) as translations, open(options.input) as input_lines:
        for input_line, translation in zip(input_lines, translations):
            src_words = input_line.strip().split(" ")
            tgt_words = translation.strip().split(" ")
            final_translation = replacer.replace(tgt_words, src_words)
            print(" ".join(final_translation))


if __name__ == "__main__":
    main()
