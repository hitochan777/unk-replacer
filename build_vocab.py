import argparse
import collections
import operator
import json
import logging
import sys
from typing import List, Tuple

from src.bpe import learn_bpe

logger = logging.getLogger(__name__)


def build_word_vocab(fn, voc_limit=None, max_nb_ex=None):
    if max_nb_ex is not None:
        logger.info("Using the first %d lines in training data" % max_nb_ex)

    f = open(fn, "r")
    counts = collections.defaultdict(int)  # type: Dict[str, int]
    for num_ex, line in enumerate(f):
        if max_nb_ex is not None and num_ex >= max_nb_ex:
            break
        line = line.rstrip().split(" ")
        for w in line:
            counts[w] += 1

    sorted_counts = sorted(counts.items(), key=operator.itemgetter(1, 0), reverse=True)
    voc = list(map(operator.itemgetter(0), sorted_counts[:voc_limit]))
    f.close()
    return voc


def build_bpe_vocab(fn, voc_limit: int, max_nb_ex: int=None) -> List[Tuple]:
    with open(fn, 'r') as f:
        rules = learn_bpe.learn(f, symbols=voc_limit, nb_lines=max_nb_ex)
        return rules


def main(args=None):
    parser = argparse.ArgumentParser(description='Build vocabulary')
    subparsers = parser.add_subparsers(dest='sub_command', help='sub-command help')
    parser_word = subparsers.add_parser('word', help='word help')
    parser_word.add_argument('--source-file', required=True, type=str, help='Training file for source language')
    parser_word.add_argument('--target-file', required=True, type=str, help='Training file for target language')
    parser_word.add_argument('--src-vocab-size', default=30000, type=int, help='Source vocabulary size')
    parser_word.add_argument('--tgt-vocab-size', default=30000, type=int, help='Target vocabulary size')
    parser_word.add_argument('--max-nb-ex', default=None, type=int, help='Max number of lines to build vocabulary')
    parser_word.add_argument('--output-file', required=True, type=str, help='Output filename')

    parser_bpe = subparsers.add_parser('bpe', help='bpe help')
    parser_bpe.add_argument('--source-file', required=True, type=str, help='Training file for source language')
    parser_bpe.add_argument('--target-file', required=True, type=str, help='Training file for target language')
    parser_bpe.add_argument('--src-vocab-size', default=20000, type=int, help='Source vocabulary size')
    parser_bpe.add_argument('--tgt-vocab-size', default=20000, type=int, help='Target vocabulary size')
    parser_bpe.add_argument('--max-nb-ex', default=None, type=int, help='Max number of lines to build vocabulary')
    parser_bpe.add_argument('--output-file', required=True, type=str, help='Output filename')

    options = parser.parse_args(args)

    if options.sub_command == "word":
        src_voc = build_word_vocab(options.source_file, voc_limit=options.src_vocab_size, max_nb_ex=options.max_nb_ex)
        tgt_voc = build_word_vocab(options.target_file, voc_limit=options.tgt_vocab_size, max_nb_ex=options.max_nb_ex)
    elif options.sub_command == "bpe":
        src_voc = build_bpe_vocab(options.source_file, voc_limit=options.src_vocab_size, max_nb_ex=options.max_nb_ex)
        tgt_voc = build_bpe_vocab(options.target_file, voc_limit=options.tgt_vocab_size, max_nb_ex=options.max_nb_ex)
    else:
        parser.print_help()
        sys.exit()

    logger.info("Writing vocabulary to %s in JSON format" % (options.output_file,))
    with open(options.output_file, 'w') as f:
        json.dump([src_voc, tgt_voc], f)

    logger.info("Finished writing")

if __name__ == '__main__':
    main()

