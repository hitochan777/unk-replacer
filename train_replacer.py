import argparse
from collections import defaultdict
from itertools import zip_longest
from itertools import groupby
import logging

from os import path
import json
from typing import Dict, List, Iterable, Union, Tuple

from src.lexical_dictionary import LexicalDictionary
from src.alignment import Alignment
from src.bpe.apply_bpe import BPE
from src.word2vec import Word2Vec

logger = logging.getLogger(__name__)


class Replacement:
    def __init__(self, src_before: str=None, src_after: str=None,
                 tgt_before=None, tgt_after=None):
        self.src_before = src_before
        self.src_after = src_after
        self.tgt_before = tgt_before
        self.tgt_after = tgt_after


class Replacer:
    def __init__(self, src_emb: Word2Vec, tgt_emb: Word2Vec,
                 lex_e2f: LexicalDictionary, lex_f2e: LexicalDictionary,
                 src_voc: Iterable[str], tgt_voc: Iterable[str],
                 src_bpe: BPE, tgt_bpe: BPE,
                 lex_prob_threshold: float=0.3, src_sim_threshold: float=0.3, tgt_sim_threshold: float=0.3) -> None:

        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.lex_e2f = lex_e2f
        self.lex_f2e = lex_f2e
        self.src_voc = src_voc
        self.tgt_voc = tgt_voc
        self.lex_prob_threshold = lex_prob_threshold
        self.src_sim_threshold = src_sim_threshold
        self.tgt_sim_threshold = tgt_sim_threshold
        self.memory = defaultdict(int)
        self.src_bpe = src_bpe  # type: BPE
        self.tgt_bpe = tgt_bpe  # type: BPE

    @staticmethod
    def is_contiguous(seq: List[int]) -> bool:
        if len(seq) == 0:
            return True

        is_sorted = all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))
        if not is_sorted:
            seq = sorted(seq)

        groups = groupby(enumerate(seq), lambda index, item: index - item)
        return len(groups) == 1

    def apply_bpe_target(self, seq: Union[str, List[str]]) -> str:
        new_seq = []
        if isinstance(seq, str):
            seq = [seq]

        for word in seq:
            if word not in self.tgt_voc:
                segmentation = self.tgt_bpe.segment_word(word)  # type: str
                new_seq.append(segmentation)
            else:
                new_seq.append(word)

        return " ".join(new_seq)

    def apply_bpe_source(self, seq: Union[str, List[str]]) -> str:
        new_seq = []
        if isinstance(seq, str):
            seq = [seq]

        for word in seq:
            if word not in self.src_voc:
                segmentation = self.src_bpe.segment_word(word)  # type: str
                new_seq.append(segmentation)
            else:
                new_seq.append(word)

        return " ".join(new_seq)

    def replace(self, src: List[str], tgt: List[str], align: Dict[int, List[int]]) -> List[str]:
        new_src_seq = list(src)  # clone src
        new_tgt_seq = list(tgt)  # clone tgt
        unk_scc = Alignment.get_scc_without_unknowns(align, src, tgt, self.src_voc, self.tgt_voc)
        for f_indices, e_indices in unk_scc:
            if len(f_indices) == 0:  # null alignment
                assert len(e_indices) == 1
                e_index = e_indices[0]
                new_tgt = self.apply_bpe_target(tgt[e_index])  # type: str
                new_tgt_seq[e_index] = new_tgt

            if len(e_indices) == 0:  # null alignment
                assert len(f_indices) == 1
                f_index = f_indices[0]
                new_src = self.apply_bpe_source(src[f_index])  # type: str
                new_src_seq[f_index] = new_src

            if len(f_indices) == 1 and len(e_indices) == 1:  # one-to-one alignment
                f_index, e_index = f_indices[0], e_indices[0]
                replacement = self.one_to_one_replace(f_index, e_index, src, tgt)
                if replacement.succeed:
                    continue

            if len(f_indices) == 1 and len(e_indices) > 1 and self.is_contiguous(e_indices):  # one-to-many
                # TODO: one-to-many replacer
                f_index = f_indices[0]
                replacement = self.one_to_many_replace(f_index, e_indices[0], e_indices[-1], src, tgt)

            if len(f_indices) > 1 and len(e_indices) == 1 and self.is_contiguous(f_indices):  # many-to-one
                # TODO: many-to-one replacer
                e_index = e_indices[0]
                replacement = self.many_to_one_replace(f_indices[0], f_indices[-1], e_index, src, tgt)

            assert len(f_indices) > 0 and len(e_indices) > 0

        # new_src = filter(None, fwords)
        # new_tgt = filter(None, ewords)
        return new_src, new_tgt

    def one_to_one_replace(self, f_index, e_index, src, tgt) -> Tuple[bool, Replacement]:
        candidates = []
        most_sim_words = self.src_embedding.most_similar_word(src[f_index])

        if src[f_index] in self.src_voc:
            most_sim_words.insert(0, (src[f_index], 1.0))

        logger.debug("most similar words of %s in the src vocab are %s" % (src[f_index], str(most_sim_words)))
        if most_sim_words[0][1] < self.src_sim_threshold:
            logger.debug("Similarity(%f) to most similar word is less than %f, so not replacing." % (most_sim_words[0][1], self.src_sim_threshold,))
            return False, None

        for most_sim_word, cos_sim in most_sim_words:
            assert most_sim_word == src[f_index] \
                   or most_sim_word in self.src_dic, "%s is not in the dictionary!" % most_sim_word
            translations = self.lexf2e.get_translations(
                most_sim_word, only_in_vocab=True,
                prob_threshold=self.lex_prob_threshold
            )
            if len(translations) > 0:
                candidates += list(
                    zip_longest([(most_sim_word, cos_sim)], translations, fillvalue=(most_sim_word, cos_sim))
                )

        if len(candidates) > 0:
            best_fword, best_eword = self.get_best_src_tgt_pair(src, tgt, f_index, e_index, candidates)
            new_fword = best_fword
            new_eword = best_eword
            if self.tgt_emb.similarity(tgt[e_index], new_eword) < self.tgt_sim_threshold:
                logger.debug(
                    "No replacement because cos(e(%s), e'(%s)) < %f" % (
                        tgt[e_index],
                        new_eword,
                        self.tgt_sim_threshold
                    )
                )
                return False, None
            else:
                pass

        return False, None

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
            align = Alignment.convert_string_to_alignment_dictionary(align_line)
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
                lex_e2f_path, lex_f2e_path, lex_topn, voc_path, src_w2v_lowercase=False,
                tgt_w2v_lowercase=False, src_bpe_code_path: str=None, tgt_bpe_code_path: str=None):

        with open(voc_path, 'r') as f:
            vocab = json.load(f)
            assert len(vocab) == 2
            src_voc = vocab[0]
            tgt_voc = vocab[1]

        lex_e2f = LexicalDictionary.read_lex_table(lex_e2f_path, topn=lex_topn)
        lex_f2e = LexicalDictionary.read_lex_table(lex_f2e_path, topn=lex_topn)
        src_emb = Word2Vec(model_path=src_w2v_model_path, topn=src_w2v_model_topn,
                           lowercase_beforehand=src_w2v_lowercase)
        tgt_emb = Word2Vec(model_path=tgt_w2v_model_path, lowercase_beforehand=tgt_w2v_lowercase)

        lex_f2e.set_vocab(tgt_voc)
        filtered_src_voc = lex_f2e.filter_vocab(src_voc)
        src_emb.set_vocab(filtered_src_voc)

        if src_bpe_code_path is not None:
            with open(src_bpe_code_path, 'r') as f:
                src_bpe = BPE(f, use_separator=True)
        else:
            src_bpe = None

        if tgt_bpe_code_path is not None:
            with open(tgt_bpe_code_path, 'r') as f:
                tgt_bpe = BPE(f, use_separator=True)
        else:
            tgt_bpe = None

        return cls(src_emb=src_emb, tgt_emb=tgt_emb, lex_e2f=lex_e2f,
                   lex_f2e=lex_f2e, src_voc=src_voc, tgt_voc=tgt_voc,
                   src_bpe=src_bpe, tgt_bpe=tgt_bpe)


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
    parser.add_argument('--source-bpe-code', default=None, type=str, help='Path to source BPE code')
    parser.add_argument('--target-bpe-code', default=None, type=str, help='Path to target BPE code')
    # TODO: add src-sim, tgt-sim, prob, threshold
    # TODO: add embedding vocab option

    options = parser.parse_args(args)

    replacer = Replacer.factory(src_w2v_model_path=options.src_w2v_model,
                                tgt_w2v_model_path=options.tgt_w2v_model,
                                src_w2v_model_topn=options.src_w2v_model_topn,
                                lex_e2f_path=options.lex_e2f,
                                lex_f2e_path=options.lex_f2e,
                                lex_topn=options.lex_topn,
                                voc_path=options.vocab,
                                src_bpe_code_path=options.src_bpe_code,
                                tgt_bpe_code_path=options.tgt_bpe_code)

    if options.train_src is not None and options.train_tgt is not None and options.train_align is not None:
        replacer.replace_parallel_corpus(options.train_src, options.train_tgt, options.train_align, options.suffix)

    if options.dev_src is not None and options.dev_tgt is not None and options.dev_align is not None:
        replacer.replace_parallel_corpus(options.dev_src, options.dev_tgt, options.dev_align, options.suffix)

if __name__ == "__main__":
    main()
