import argparse
from collections import defaultdict
from typing import List
from dotmap import DotMap
from itertools import zip_longest
from os import path
import json


class Replacer:
    def __init__(self, src_emb, tgt_emb, lex_e2f, lex_f2e, src_voc, tgt_voc):
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.lex_e2f = lex_e2f
        self.lex_f2e = lex_f2e
        self.src_voc = src_voc
        self.tgt_voc = tgt_voc
        self.memory = defaultdict(int)

    def replace(self, src: List[str], tgt: List[str], align) -> List[str]:
        return []

    def replace_parallel_corpus(self, src_file: str, tgt_file: str, align_file: str, suffix: str):
        new_src_filename = src_file + suffix
        new_tgt_filename = tgt_file + suffix
        existing_files = []
        if path.isfile(new_src_filename):
            existing_files.append(new_src_filename)

        if path.isfile(new_tgt_filename):
            existing_files.append(new_tgt_filename)

        if len(existing_files) > 0:
            print("%s will be overwritten. Press Enter to proceed." % (" ".join(existing_files),))

        new_src_file = open(new_src_filename, 'r')
        new_tgt_file = open(new_tgt_filename, 'r')
        src_lines = open(src_file, 'r')
        tgt_lines = open(tgt_file, 'r')
        align_lines = open(align_file, 'r')

        for src_line, tgt_line, align_line in zip_longest(src_lines, tgt_lines, align_lines):
            if src_line is None or tgt_line is None or align_line is None:
                break

            src_tokens = src_line.strip().split(' ')
            tgt_tokens = tgt_line.strip().split(' ')
            align = None  # TODO: get alignment
            new_src_tokens, new_tgt_tokens = self.replace(src_tokens, tgt_tokens, align)
            print(" ".join(new_src_tokens), file=new_src_file)
            print(" ".join(new_tgt_tokens), file=new_tgt_file)

        # close file streams
        src_lines.close()
        tgt_lines.close()
        align_lines.close()
        new_src_file.close()
        new_tgt_file.close()

    @classmethod
    def factory(cls, src_w2v_model_path, tgt_w2v_model_path, src_w2v_model_topn,
                         lex_e2f_path, lex_f2e_path, lex_topn, voc_path):
        components = DotMap()
    
        with open(src_w2v_model_path, 'r') as f:
            pass
    
        with open(tgt_w2v_model_path, 'r') as f:
            pass
    
        with open(lex_e2f_path, 'r') as f:
            pass
    
        with open(lex_f2e_path, 'r') as f:
            pass
    
        with open(voc_path, 'r') as f:
            vocab = json.load(f)
            assert len(vocab) == 2
            src_voc = vocab[0]
            tgt_voc = vocab[1]

        return cls(src_emb=components.src_emb, tgt_emb=components.tgt_emb, lex_e2f=components.lex_e2f,
                   lex_f2e=components.lex_f2e, src_voc=components.src_voc, tgt_voc=components.tgt_voc)


def main(args=None):
    parser = argparse.ArgumentParser(description='Replace training data')
    parser.add_argument('--src-w2v-model', required=True, type=str, help='Path to source word2vec model')
    parser.add_argument('--tgt-w2v-model', required=True, type=str, help='Path to target word2vec model')
    parser.add_argument('--src-w2v-model-topn', metavar='K', default=10, type=int,
                        help='Use top %(metavar)s most similar words from source word2vec')
    parser.add_argument('--lex-e2f', required=True, type=str, help='Path to target to source lexical dictionary')
    parser.add_argument('--lex-f2e', required=True, type=str, help='Path to source to source lexical dictionary')
    parser.add_argument('--lex-topn', metavar='K', default=None, type=int,
                        help='Use the first %(metavar)s translations with the highest probability in lexical dictionary')
    parser.add_argument('--train-src', default=None, type=str, help='Path to source training data')
    parser.add_argument('--train-tgt', default=None, type=str, help='Path to target training data')
    parser.add_argument('--dev-src', default=None, type=str, help='Path to source dev data')
    parser.add_argument('--dev-tgt', default=None, type=str, help='Path to target dev data')
    parser.add_argument('--train-align', required=True, type=str, default=None, help='Path to word alignment file for training data')
    parser.add_argument('--dev-align', required=True, type=str, default=None, help='Path to word alignment file for dev data')
    parser.add_argument('--replaced_suffix', required=True, type=str,
                        help='Suffix for newly created training and dev data')
    parser.add_argument('--vocab', required=True, type=str, help='Path to vocabulary file')
    # TODO: add src-sim, tgt-sim, prob, threshold
    # TODO: add embedding vocab option

    options = parser.parse_args(args)

    replacer = Replacer.factory(src_w2v_model_path=options.src_w2v_model,
                                tgt_w2v_model_path=options.tgt_w2v_model,
                                src_w2v_model_topn=options.src_w2v_model_topn,
                                lex_e2f_path=options.lex_e2f,
                                lex_f2e_path=options.lex_f2e,
                                lex_topn=options.lex_topn,
                                voc_path=options.vocab)

    if options.train_src is not None and options.train_tgt is not None and options.train_align is not None:
        replacer.replace_parallel_corpus(options.train_src, options.train_tgt, options.train_align, options.suffix)

    if options.dev_src is not None and options.dev_tgt is not None and options.dev_align is not None:
        replacer.replace_parallel_corpus(options.dev_src, options.dev_tgt, options.dev_align, options.suffix)

if __name__ == "__main__":
    main()
