import argparse
import logging
from itertools import product


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_corpora(input_filename, prefix, lang):
    langs = lang.split("-")
    assert len(langs) == 2
    l1, l2 = langs
    l1_words = []
    l2_words = []
    l1_filename = "%s.%s" % (prefix, l1)
    l2_filename = "%s.%s" % (prefix, l2)

    logger.info("Corpus for %s will be written to %s" % (l1, l1_filename))
    logger.info("Corpus for %s will be written to %s" % (l2, l2_filename))
    input("Press Enter to continue.")

    with open(input_filename, 'r') as lines, open(l1_filename, 'w') as l1_fs, open(l2_filename, 'w') as l2_fs: 
        for line in lines:
            if line.startswith("#"):
                if len(l1_words) == 0 and len(l2_words) == 0:
                    continue

                if len(l1_words) == 0 or len(l2_words) == 0:
                    continue
                        
                for l1_word, l2_word in product(l1_words, l2_words):
                    print(l1_word, file=l1_fs)
                    print(l2_word, file=l2_fs)

                l1_words = []
                l2_words = []

            elif line.startswith(l1):
                word = line.lstrip().split(' ')[1]
                word = word.strip()
                if word == "":
                    logger.warning("Empty word! Skipping!")
                    continue

                l1_words.append(word)
            elif line.startswith(l2):
                word = line.lstrip().split(' ')[1]
                word = word.strip()
                if word == "":
                    logger.warning("Empty word! Skipping!")
                    continue

                l2_words.append(word)

        if len(l1_words) > 0 and len(l2_words) > 0:
            for l1_word, l2_word in product(l1_words, l2_words):
                print(l1_word, file=l1_fs)
                print(l2_word, file=l2_fs)


def main(args=None):
    parser = argparse.ArgumentParser(description='Extract parallel corpora from dictionary')
    parser.add_argument('input', type=str, help='Path to dictionary file')
    parser.add_argument('lang', help='Language pair e.g., ja-en. Order does not matter')
    parser.add_argument('prefix', metavar='PREFIX', help='Prefix of output file. If lang is l1-l2, corpora for the languages are written to %(metavar)s.l1 and %(metavar)s.l2 respectively.')

    options = parser.parse_args(args)
    
    extract_corpora(options.input, options.prefix, options.lang)

if __name__ == "__main__":
    main()
