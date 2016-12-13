import argparse
from collections import defaultdict
from collections import namedtuple
from itertools import zip_longest
from itertools import groupby
import logging
import sys

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
                 lex_prob_threshold: float=0.3, src_sim_threshold: float=0.5, tgt_sim_threshold: float=0.5, allow_unk_character: bool=True) -> None:

        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.lex_e2f = lex_e2f
        self.lex_f2e = lex_f2e
        self.src_voc = src_voc
        self.tgt_voc = tgt_voc
        logger.info("lex_prob_threshold: %f" % lex_prob_threshold)
        self.lex_prob_threshold = lex_prob_threshold
        logger.info("src_sim_threshold: %f" % src_sim_threshold)
        self.src_sim_threshold = src_sim_threshold
        logger.info("tgt_sim_threshold: %f" % tgt_sim_threshold)
        self.tgt_sim_threshold = tgt_sim_threshold
        self.memory = defaultdict(int)
        self.src_bpe = src_bpe  # type: BPE
        self.tgt_bpe = tgt_bpe  # type: BPE

        self.allow_unk_character = allow_unk_character

    def set_allow_unk_character(self, allow_unk_character):
        self.allow_unk_character = allow_unk_character

    def export_memory(self, output: str) -> None:
        if path.isfile(output):
            input("%s will be overwritten. Press Enter to proceed." % (output,))

        logger.info("Writing replacement of size %d to %s... Wait patiently..." % (len(self.memory), output))
        with open(output, "w") as f:
            memory = []
            for mem, freq in self.memory.items():
                memory.append([list(mem), freq])

            # print(memory)
            json.dump(memory, f)  # json.dump does not work with defaultdict

        logger.info("Finished exporting memory")

    @staticmethod
    def is_contiguous(seq: List[int]) -> bool:
        if len(seq) == 0:
            return True

        is_sorted = all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))
        if not is_sorted:
            seq = sorted(seq)

        groups = groupby(enumerate(seq), lambda args: args[0] - args[1])

        return len(list(groups)) == 1

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
                assert self.allow_unk_character or "<unk>" not in new_tgt, "In null-to-one: new target sequence %s contains <unk>" % new_tgt
                new_tgt_seq[e_index] = new_tgt
                continue

            if len(e_indices) == 0:  # null alignment
                assert len(f_indices) == 1
                f_index = f_indices[0]
                new_src = self.apply_bpe_source(src[f_index])  # type: str
                assert self.allow_unk_character or "<unk>" not in new_src, "In one-to-null: New source sequence %s contains <unk>" % new_src
                new_src_seq[f_index] = new_src
                if "<unk>" not in new_src:
                    self.memory[Replacement(src[f_index], "<null>", new_src, "<null>")] += 1
                continue

            if len(f_indices) == 1 and len(e_indices) == 1:  # one-to-one alignment
                f_index, e_index = f_indices[0], e_indices[0]
                success, (new_src, new_tgt) = self.one_to_one_replace(src[f_index], tgt[e_index])

                new_src_seq[f_index] = new_src
                new_tgt_seq[e_index] = new_tgt
                if success:
                    self.memory[Replacement(src[f_index], tgt[e_index], new_src, new_tgt)] += 1
                    continue

            if len(f_indices) == 1 and len(e_indices) > 1 and self.is_contiguous(e_indices):  # one-to-many
                f_index = f_indices[0]
                success, (new_src, new_tgt) = self.one_to_many_replace(src[f_index], tgt[e_indices[0]:e_indices[-1]+1])
                new_src_seq[f_index] = new_src
                new_tgt_seq[e_indices[0]:e_indices[-1]+1] = [None] * len(e_indices)
                new_tgt_seq[e_indices[0]] = new_tgt
                orig_tgt_string = " ".join(tgt[e_indices[0]:e_indices[-1]+1])
                if success:
                    self.memory[Replacement(src[f_index], orig_tgt_string, new_src, new_tgt)] += 1
                    continue

            if len(f_indices) > 1 and len(e_indices) == 1 and self.is_contiguous(f_indices):  # many-to-one
                e_index = e_indices[0]
                success, (new_src, new_tgt) = self.many_to_one_replace(src[f_indices[0]:f_indices[-1]+1], tgt[e_index])
                new_src_seq[f_indices[0]:f_indices[-1]+1] = [None] * len(f_indices)
                new_src_seq[f_indices[0]] = new_src
                new_tgt_seq[e_index] = new_tgt
                orig_src_string = " ".join(src[f_indices[0]:f_indices[-1]+1])
                if success:
                    self.memory[Replacement(orig_src_string, tgt[e_index], new_src, new_tgt)] += 1
                    continue

            for index in f_indices:
                new_src = self.apply_bpe_source(src[index])
                assert self.allow_unk_character or "<unk>" not in new_src, "New source sequence %s contains <unk>" % new_src
                new_src_seq[index] = new_src

            for index in e_indices:
                new_tgt = self.apply_bpe_target(tgt[index])
                assert self.allow_unk_character or "<unk>" not in new_tgt, "New target sequence %s contains <unk>" % new_tgt
                new_tgt_seq[index] = new_tgt

            if self.is_contiguous(f_indices) and self.is_contiguous(e_indices):  # save in memory
                orig_src_string = " ".join(src[f_indices[0]:f_indices[-1]+1])
                orig_tgt_string = " ".join(tgt[e_indices[0]:e_indices[-1]+1])
                new_src_string = " ".join(new_src_seq[f_indices[0]:f_indices[-1]+1])
                new_tgt_string = " ".join(new_tgt_seq[e_indices[0]:e_indices[-1]+1])
                self.memory[Replacement(orig_src_string, orig_tgt_string, new_src_string, new_tgt_string)] += 1

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
            assert isinstance(src_word, str) and isinstance(tgt_word, str)
            lex_prob = candidate.lex_prob

            if len(src) == 1 and len(tgt) == 1:
                src_sim = candidate.cos_sim
                tgt_sim = self.tgt_emb.similarity(tgt_word, tgt[0])
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
        most_sim_words = self.src_emb.most_similar_word(src)
        if len(most_sim_words) == 0:
            return False, (None, None)

        logger.debug("most similar words of %s in the src vocab are %s" % (src[0], str(most_sim_words)))
        if most_sim_words[0].similarity < self.src_sim_threshold:
            logger.debug("Similarity(%f) to most similar word is less than %f, so not replacing." % (most_sim_words[0][1], self.src_sim_threshold,))
            return False, (None, None)

        for most_sim_word, cos_sim in most_sim_words:
            assert most_sim_word == src or most_sim_word in self.src_voc, "%s is not in the dictionary!" % most_sim_word
            translations = self.lex_f2e.get_translations(
                most_sim_word, only_in_vocab=True,
                prob_threshold=self.lex_prob_threshold
            )
            for target_word, prob in translations:
                candidates.append(Candidate(most_sim_word, target_word, cos_sim, prob))

        if len(candidates) > 0:
            best_src_word, best_tgt_word = self.get_best_candidate([src], tgt, candidates)
            assert best_src_word in self.src_voc and best_tgt_word in self.tgt_voc
            if len(tgt) == 1 and self.tgt_emb.similarity(tgt[0], best_tgt_word) < self.tgt_sim_threshold:
                logger.debug(
                    "No replacement because cos(e(%s), e'(%s)) < %f" % (
                        tgt[0],
                        best_tgt_word,
                        self.tgt_sim_threshold
                    )
                )
                return False, (None, None)
            else:
                return True, (best_src_word, best_tgt_word)

        return False, (None, None)

    def many_to_one_replace(self, src: List[str], tgt: str) -> Tuple[bool, Tuple[str, str]]:
        candidates = []
        most_sim_words = self.tgt_emb.most_similar_word(tgt)
        if len(most_sim_words) == 0:
            return False, (None, None)

        logger.debug("most similar words of %s in the tgt vocab are %s" % (tgt, str(most_sim_words)))
        if most_sim_words[0].similarity < self.tgt_sim_threshold:
            logger.debug("Similarity(%f) to most similar word is less than %f, so not replacing." % (most_sim_words[0][1], self.tgt_sim_threshold,))
            return False, (None, None)

        for most_sim_word, cos_sim in most_sim_words:
            assert most_sim_word == src or most_sim_word in self.tgt_voc, "%s is not in the dictionary!" % most_sim_word
            translations = self.lex_e2f.get_translations(
                most_sim_word, only_in_vocab=True,
                prob_threshold=self.lex_prob_threshold
            )
            for source_word, prob in translations:
                candidates.append(Candidate(source_word, most_sim_word, cos_sim, prob))

        if len(candidates) > 0:
            best_src_word, best_tgt_word = self.get_best_candidate(src, [tgt], candidates)
            assert best_src_word in self.src_voc and best_tgt_word in self.tgt_voc
            if len(src) == 1 and self.src_emb.similarity(src[0], best_src_word) < self.src_sim_threshold:
                logger.debug(
                    "No replacement because cos(e(%s), e'(%s)) < %f" % (
                        src[0],
                        best_src_word,
                        self.src_sim_threshold
                    )
                )
                return False, (None, None)
            else:
                return True, (best_src_word, best_tgt_word)

        return False, (None, None)

    def replace_parallel_corpus(self, src_file: str, tgt_file: str, align_file: str, suffix: str, print_per_lines: int=10000):
        new_src_filename = src_file + suffix
        new_tgt_filename = tgt_file + suffix
        existing_files = []
        if path.isfile(new_src_filename):
            existing_files.append(new_src_filename)

        if path.isfile(new_tgt_filename):
            existing_files.append(new_tgt_filename)

        if len(existing_files) > 0:
            input("%s will be overwritten. Press Enter to proceed." % (" ".join(existing_files),))

        new_src_file = open(new_src_filename, 'w')
        new_tgt_file = open(new_tgt_filename, 'w')
        src_lines = open(src_file, 'r')
        tgt_lines = open(tgt_file, 'r')
        align_lines = open(align_file, 'r')

        for index, (src_line, tgt_line, align_line) in enumerate(zip_longest(src_lines, tgt_lines, align_lines)):

            if src_line is None or tgt_line is None or align_line is None:
                break

            src_tokens = src_line.strip().split(' ')
            tgt_tokens = tgt_line.strip().split(' ')
            align = Alignment.convert_string_to_alignment_dictionary(align_line)
            try:
                new_src_tokens, new_tgt_tokens = self.replace(src_tokens, tgt_tokens, align)
            except Exception as e:
                print("Failed at line %d." % (index + 1,), file=sys.stderr)
                print(e)
                sys.exit()

            print(" ".join(new_src_tokens), file=new_src_file)
            print(" ".join(new_tgt_tokens), file=new_tgt_file)
            if (index + 1) % print_per_lines == 0:
                print("Finished processing %d lines" % (index + 1))

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
        logger.info("Loading vocabulary from %s" % voc_path)
        with open(voc_path, 'r') as f:
            vocab = json.load(f)
            assert len(vocab) == 2
            src_voc = vocab[0]
            tgt_voc = vocab[1]

        
        logger.info("Loading e2f lexical dictionary from %s" % lex_e2f_path)
        lex_e2f = LexicalDictionary.read_lex_table(lex_e2f_path, topn=lex_topn)
        logger.info("Loading f2e lexical dictionary from %s" % lex_f2e_path)
        lex_f2e = LexicalDictionary.read_lex_table(lex_f2e_path, topn=lex_topn)

        logger.info("Loading source word2vec model from %s" % src_w2v_model_path)
        src_emb = Word2Vec(model_path=src_w2v_model_path, topn=src_w2v_model_topn,
                           lowercase_beforehand=src_w2v_lowercase)
        logger.info("Loading target word2vec model from %s" % tgt_w2v_model_path)
        tgt_emb = Word2Vec(model_path=tgt_w2v_model_path, lowercase_beforehand=tgt_w2v_lowercase)

        logger.info("Setting vocabulary of f2e lexical dictionary")
        lex_f2e.set_vocab(tgt_voc)

        logger.info("Setting vocabulary of e2f lexical dictionary")
        lex_e2f.set_vocab(src_voc)

        logger.info("Filtering source vocabulary")
        filtered_src_voc = lex_f2e.filter_vocab(src_voc)

        logger.info("Filtering target vocabulary")
        filtered_tgt_voc = lex_e2f.filter_vocab(tgt_voc)

        logger.info("Setting vocabulary in source word2vec")
         
        src_emb.set_vocab(filtered_src_voc)

        logger.info("Setting vocabulary in target word2vec")
        tgt_emb.set_vocab(filtered_tgt_voc)

        if src_bpe_code_path is not None:
            logger.info("Loading source BPE codes from %s" % src_bpe_code_path)
            with open(src_bpe_code_path, 'r') as f:
                src_bpe = BPE(f, use_separator=True)
                logger.info("Vocab size of source BPE: %d", len(src_bpe.get_vocab()))
        else:
            src_bpe = None

        if tgt_bpe_code_path is not None:
            logger.info("Loading target BPE codes from %s" % tgt_bpe_code_path)
            with open(tgt_bpe_code_path, 'r') as f:
                tgt_bpe = BPE(f, use_separator=True)
                logger.info("Vocab size of target BPE: %d", len(tgt_bpe.get_vocab()))
        else:
            tgt_bpe = None
        
        logger.info("Building Replacer instance")
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
    parser.add_argument('--replaced-suffix', required=True, type=str,
                        help='Suffix for newly created training and dev data')
    parser.add_argument('--vocab', required=True, type=str, help='Path to vocabulary file')
    parser.add_argument('--src-bpe-code', default=None, type=str, help='Path to source BPE code')
    parser.add_argument('--tgt-bpe-code', default=None, type=str, help='Path to target BPE code')
    parser.add_argument('--memory', required=True, type=str, help='Save path to replacement memory')
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
        logger.info("Processing training data")
        replacer.set_allow_unk_character(False)
        replacer.replace_parallel_corpus(options.train_src, options.train_tgt, options.train_align, options.replaced_suffix, print_per_lines=10000)

    if options.dev_src is not None and options.dev_tgt is not None and options.dev_align is not None:
        logger.info("Processing dev data")
        replacer.set_allow_unk_character(True)
        replacer.replace_parallel_corpus(options.dev_src, options.dev_tgt, options.dev_align, options.replaced_suffix, print_per_lines=100)

    logger.info("Finally writing replacement memory")
    replacer.export_memory(options.memory)

if __name__ == "__main__":
    main()
