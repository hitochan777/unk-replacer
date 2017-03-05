import argparse
import sys
import logging

# this is for suppressing unnecessary logging by gensim modeule
logging.getLogger("summa.preprocessing.cleaner").setLevel(logging.WARNING)

from unk_replacer.bin import train_replacer
from unk_replacer.bin import test_replacer
from unk_replacer.bin import restorer
from unk_replacer.bin import jean_replace
from unk_replacer.bin import build_vocab
from unk_replacer.bin import combine_bpe_word_vocab
from unk_replacer import version


def main(arguments=None):
    # create the top-level parser
    parser = argparse.ArgumentParser(description="Replacer: Unknown Word Processor for Neural Machine Translation with Attention Mechanism", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     prog = "unk-rep")


    subparsers = parser.add_subparsers(dest="__subcommand_name")
    subparsers.required = True

    # create the parser for the "replace_parallel" command
    parser_replace_parallel = subparsers.add_parser('replace-parallel', description= "Replace unknown words in parallel corpora", help="Replace unknown words in parallel corpora",
                                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train_replacer.define_parser(parser_replace_parallel)

    # create the parser for the "replace_input" command
    parser_replace_input = subparsers.add_parser('replace-input', description= "Replace unknown words in input sentences", help="Replace unknown words in input sentences", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    test_replacer.define_parser(parser_replace_input)

    # create the parser for the "restore" command
    parser_restore = subparsers.add_parser('restore', description= "Restore the final translation", help="Restore the final translation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    restorer.define_parser(parser_restore)

    # create the parser for the "build-vocab" command
    parser_build_vocab = subparsers.add_parser('build-vocab', description= "Build vocabulary from a parallel corpus", help="Build vocabulary from a parallel corpus", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    build_vocab.define_parser(parser_build_vocab)

    # create the parser for the "combine-word-and-bpe-vocab" command
    parser_combine_vocab = subparsers.add_parser('combine-word-and-bpe-vocab', description= "Build", help="Combine word and BPE vocab", formatter_class=argparse.ArgumentDefaultsHelpFormatter, aliases=['co'])
    combine_bpe_word_vocab.define_parser(parser_combine_vocab)


    # create the parser for the "version" command
    parser_version = subparsers.add_parser('version', description= "Get version infos.", help = "Get version infos", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    options = parser.parse_args(args=arguments)

    func_dict = {"build_vocab": build_vocab.run,
                 "replace-parallel": train_replacer.run,
                 "replace-input": test_replacer.run,
                 "restore":  restorer.run,
                 "version": version.main,
                 "build-vocab": build_vocab.run,
                 "combine-word-and-bpe-vocab": combine_bpe_word_vocab.run
                 }

    func = func_dict[options.__subcommand_name]
    func(options)

if __name__ == "__main__":
    main()
