import argparse
from collections import defaultdict
from collections import namedtuple
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

Candidate = namedtuple('Candidate', 'src_word tgt_word cos_sim lex_prob')
Replacement = namedtuple('Replacement', 'src_before tgt_before src_after tgt_after')


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

    def export_memory(self, output: str) -> None:
        if path.isfile(output):
            print("%s will be overwritten. Press Enter to proceed." % (output,))

        logger.info("Writing replacement of size %d to %s... Wait patiently..." % (len(self.memory), output))
        with open(output, "w") as f:
            json.dump(f, self.memory)

        logger.info("Finished exporting memory")

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
                continue

            if len(e_indices) == 0:  # null alignment
                assert len(f_indices) == 1
                f_index = f_indices[0]
                new_src = self.apply_bpe_source(src[f_index])  # type: str
                new_src_seq[f_index] = new_src
                self.memory[Replacement(src[0], None, new_src, None)] += 1
                continue

            if len(f_indices) == 1 and len(e_indices) == 1:  # one-to-one alignment
                f_index, e_index = f_indices[0], e_indices[0]
                success, (new_src, new_tgt) = self.one_to_one_replace(src[f_index], tgt[e_index])
                new_src_seq[f_index] = new_src
                new_tgt_seq[e_index] = new_tgt
                self.memory[Replacement(src[f_index], tgt[e_index], new_src, new_tgt)] += 1
                if success:
                    continue

            if len(f_indices) == 1 and len(e_indices) > 1 and self.is_contiguous(e_indices):  # one-to-many
                f_index = f_indices[0]
                success, (new_src, new_tgt) = self.one_to_many_replace(src[f_index], tgt[e_indices[0]:e_indices[-1]+1])
                new_src_seq[f_index] = new_src
                new_tgt_seq[e_indices[0]:e_index[-1]+1] = [None] * len(e_indices)
                new_tgt_seq[e_indices[0]] = new_tgt
                orig_tgt_string = " ".join(tgt[e_indices[0]:e_indices[-1]+1])
                self.memory[Replacement(src[f_index], orig_tgt_string, new_src, new_tgt)] += 1
                if success:
                    continue

            if len(f_indices) > 1 and len(e_indices) == 1 and self.is_contiguous(f_indices):  # many-to-one
                e_index = e_indices[0]
                success, replacement = self.many_to_one_replace(f_indices[0], f_indices[-1], e_index, src, tgt)
                new_src_seq[f_indices[0]:f_indices[-1]+1] = [None] * len(f_indices)
                new_src_seq[f_indices[0]] = new_src
                new_tgt_seq[e_index] = new_tgt
                orig_src_string = " ".join(src[f_indices[0]:f_indices[-1]+1])
                self.memory[Replacement(orig_src_string, tgt[e_index], new_src, new_tgt)] += 1
                if success:
                    continue

            if len(f_indices) > 1 and len(e_indices) > 1:  # many-to-many
                if self.is_contiguous(f_indices) and self.is_contiguous(e_indices):
                    new_src = self.apply_bpe_source(src[f_indices[0]:f_indices[-1]+1])
                    new_tgt = self.apply_bpe_target(tgt[e_indices[0]:e_indices[-1]+1])
                    new_src_seq[f_indices[0]:f_indices[-1]+1] = [None] * len(f_indices)
                    new_src_seq[f_indices[0]] = new_src
                    new_tgt_seq[e_indices[0]:e_indices[-1]+1] = [None] * len(e_indices)
                    new_tgt_seq[e_indices[0]] = new_tgt
                    # TODO: save in memory
                else:
                    pass
            else:  # garbage collector
                if self.is_contiguous(f_indices) and self.is_contiguous(e_indices):
                    # TODO: save in memory
                    pass
                else:
                    pass

        new_src_seq = list(filter(None, new_src_seq))
        new_tgt_seq = list(filter(None, new_tgt_seq))
        return new_src_seq, new_tgt_seq

    def get_best_candidate(self, src: List[str], tgt: List[str], candidates: List[Candidate]) -> Tuple[str, str]:
        """
        candidates must be sorted by cos_sim, and then lex_prob in the descending order
        """
        best_pair = None
        best_score = 0.0
        assert not(len(src) > 1 and len(tgt) > 1), "At least one side has only one word"
        for candidate in candidates:
            src_word, tgt_word = candidate.src_word, candidate.tgt_word
            assert len(src_word) == 1 and len(tgt_word) == 1
            lex_prob = candidate.lex_prob

            if len(src) == 1 and len(tgt) == 1:
                src_sim = candidate.cos_sim
                tgt_sim = self.tgt_embedding.similarity(tgt_word, tgt[0])
                score = (src_sim + tgt_sim) * lex_prob

            elif len(src) == 1 and len(tgt) > 1:
                src_sim = candidate.cos_sim
                score = src_sim * lex_prob

            elif len(src) > 1 and len(tgt) == 1:
                tgt_sim = candidate.cos_sim
                score = tgt_sim * lex_prob
            else:
                raise RuntimeError("Unexpected Error")

            if score > best_score:
                best_pair = (src_word, tgt_word)
                best_score = score

        return best_pair

    def one_to_one_replace(self, src: str, tgt: str) -> Tuple[bool, Replacement]:
        return self.one_to_many_replace(src, [tgt])

    def one_to_many_replace(self, src: str, tgt: List[str]) -> Tuple[bool, Tuple[str, str]]:
        candidates = []
        most_sim_words = self.src_embedding.most_similar_word(src)

        logger.debug("most similar words of %s in the src vocab are %s" % (src[0], str(most_sim_words)))
        if most_sim_words[0].similarity < self.src_sim_threshold:
            logger.debug("Similarity(%f) to most similar word is less than %f, so not replacing." % (most_sim_words[0][1], self.src_sim_threshold,))
            return False, None

        for most_sim_word, cos_sim in most_sim_words:
            assert most_sim_word == src or most_sim_word in self.src_dic, "%s is not in the dictionary!" % most_sim_word
            translations = self.lexf2e.get_translations(
                most_sim_word, only_in_vocab=True,
                prob_threshold=self.lex_prob_threshold
            )
            for target_word, prob in translations:
                candidates.append(Candidate(most_sim_word, target_word, cos_sim, prob))

        if len(candidates) > 0:
            best_src_word, best_tgt_word = self.get_best_candidate([src], tgt, candidates)
            if len(tgt) == 1 and self.tgt_emb.similarity(tgt[0], best_tgt_word) < self.tgt_sim_threshold:
                logger.debug(
                    "No replacement because cos(e(%s), e'(%s)) < %f" % (
                        tgt[0],
                        best_tgt_word,
                        self.tgt_sim_threshold
                    )
                )
                return False, None
            else:
                return True, (best_src_word, best_tgt_word)

        return False, None

    def many_to_one_replace(self, src: List[str], tgt: str) -> Tuple[bool, Tuple[str, str]]:
        candidates = []
        most_sim_words = self.tgt_embedding.most_similar_word(tgt)

        logger.debug("most similar words of %s in the tgt vocab are %s" % (tgt, str(most_sim_words)))
        if most_sim_words[0].similarity < self.tgt_sim_threshold:
            logger.debug("Similarity(%f) to most similar word is less than %f, so not replacing." % (most_sim_words[0][1], self.src_tgt_threshold,))
            return False, None

        for most_sim_word, cos_sim in most_sim_words:
            assert most_sim_word == src or most_sim_word in self.tgt_dic, "%s is not in the dictionary!" % most_sim_word
            translations = self.lexe2f.get_translations(
                most_sim_word, only_in_vocab=True,
                prob_threshold=self.lex_prob_threshold
            )
            for target_word, prob in translations:
                candidates.append(Candidate(most_sim_word, target_word, cos_sim, prob))

        if len(candidates) > 0:
            best_src_word, best_tgt_word = self.get_best_candidate(src, [tgt], candidates)
            if len(src) == 1 and self.src_emb.similarity(src[0], best_src_word) < self.src_sim_threshold:
                logger.debug(
                    "No replacement because cos(e(%s), e'(%s)) < %f" % (
                        src[0],
                        best_src_word,
                        self.src_sim_threshold
                    )
                )
                return False, None
            else:
                return True, (best_src_word, best_tgt_word)

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
