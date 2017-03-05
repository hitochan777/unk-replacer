import argparse
from collections import defaultdict
from collections import namedtuple
from itertools import zip_longest
from itertools import groupby
import logging
import sys
import textwrap

import os
import json
from typing import Dict, List, Iterable, Union, Tuple

from unk_replacer.lexical_dictionary import LexicalDictionary
from unk_replacer.alignment import Alignment
from unk_replacer.bpe.apply_bpe import BPE
from unk_replacer.word2vec import Word2Vec
from unk_replacer.number_normalizer import NumberHandler

logger = logging.getLogger(__name__)

Candidate = namedtuple('Candidate', 'src_word tgt_word cos_sim lex_prob')
Replacement = namedtuple('Replacement', 'src_before tgt_before src_after tgt_after')


class Replacer:
    def __init__(self, replace_type: str, store_memory: bool, src_emb: Word2Vec, tgt_emb: Word2Vec,
                 lex_e2f: LexicalDictionary, lex_f2e: LexicalDictionary,
                 src_voc: Iterable[str], tgt_voc: Iterable[str],
                 src_bpe: BPE, tgt_bpe: BPE,
                 lex_prob_threshold: float=0.3, src_sim_threshold: float=0.5, tgt_sim_threshold: float=0.5,
                 allow_unk_character: bool=True, backoff: str="unk", handle_numbers: bool=False) -> None:

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

        logger.info("Using %s as a backoff method" % backoff)
        self.backoff = backoff

        logger.info("Replace type is %s" % replace_type)
        self.replace_type = replace_type

        logger.info("Memorization of replacement is %s" % ("ON" if store_memory else "OFF"))
        self.store_memory = store_memory

        logger.info('Applying special handling to numbers')
        self.handle_numbers = handle_numbers

        if handle_numbers:
            assert backoff == 'unk'

    def set_allow_unk_character(self, allow_unk_character):
        self.allow_unk_character = allow_unk_character

    def export_memory(self, output: str) -> None:
        if os.path.isfile(output):
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
        unk_scc = Alignment.get_scc_with_unknowns(align, src, tgt, self.src_voc, self.tgt_voc)
        if self.handle_numbers:
            new_src = list(map(lambda word: NumberHandler.process_number(word), src))
            new_tgt = list(map(lambda word: NumberHandler.process_number(word), tgt))
            assert len(new_src) == len(src)
            assert len(new_tgt) == len(tgt)
            src_changed = [new_src[i] != src[i] for i in range(len(src))]
            tgt_changed = [new_tgt[i] != tgt[i] for i in range(len(tgt))]
            src = new_src
            tgt = new_tgt

        new_src_seq = list(src)  # clone src
        new_tgt_seq = list(tgt)  # clone tgt

        for f_indices, e_indices in unk_scc:
            if self.handle_numbers:
                contain_numbers = False
                for f_index in f_indices:
                    if src_changed[f_index]:
                        contain_numbers = True
                        break

                for e_index in e_indices:
                    if tgt_changed[e_index]:
                        contain_numbers = True
                        break

                if contain_numbers:
                    continue

            if len(f_indices) == 0:  # null alignment
                assert len(e_indices) == 1
                if self.backoff == "unk":
                    continue
                elif self.backoff == "bpe":
                    e_index = e_indices[0]
                    new_tgt = self.apply_bpe_target(tgt[e_index])  # type: str
                    assert self.allow_unk_character or "<unk>" not in new_tgt, "In null-to-one: new target sequence %s contains <unk>" % new_tgt
                    new_tgt_seq[e_index] = new_tgt
                    continue
                else:
                    raise NotImplementedError()

            if len(e_indices) == 0:  # null alignment
                assert len(f_indices) == 1
                f_index = f_indices[0]
                if self.backoff == "unk":
                    if self.store_memory:
                        self.memory[Replacement(src[f_index], "<null>", "<@UNK>", "<null>")] += 1
                    new_src_seq[f_index] = "<@UNK>"
                    continue
                elif self.backoff == "bpe":
                    new_src = self.apply_bpe_source(src[f_index])  # type: str
                    assert self.allow_unk_character or "<unk>" not in new_src, "In one-to-null: New source sequence %s contains <unk>" % new_src
                    new_src_seq[f_index] = new_src
                    if "<unk>" not in new_src:
                        if self.store_memory:
                            self.memory[Replacement(src[f_index], "<null>", new_src, "<null>")] += 1

                    continue
                else:
                    raise NotImplementedError()

            if len(f_indices) == 1 and len(e_indices) == 1:  # one-to-one alignment
                f_index, e_index = f_indices[0], e_indices[0]
                success, (new_src, new_tgt) = self.one_to_one_replace(src[f_index], tgt[e_index])

                if success:
                    new_src_seq[f_index] = new_src
                    new_tgt_seq[e_index] = new_tgt
                    if self.store_memory:
                        self.memory[Replacement(src[f_index], tgt[e_index], new_src, new_tgt)] += 1
                    continue

            if self.replace_type == "multi" and len(f_indices) == 1 and len(e_indices) > 1 and self.is_contiguous(e_indices):  # one-to-many
                f_index = f_indices[0]
                success, (new_src, new_tgt) = self.one_to_many_replace(src[f_index], tgt[e_indices[0]:e_indices[-1]+1])
                if success:
                    new_src_seq[f_index] = new_src
                    new_tgt_seq[e_indices[0]:e_indices[-1]+1] = [None] * len(e_indices)
                    new_tgt_seq[e_indices[0]] = new_tgt
                    orig_tgt_string = " ".join(tgt[e_indices[0]:e_indices[-1]+1])
                    if self.store_memory:
                        self.memory[Replacement(src[f_index], orig_tgt_string, new_src, new_tgt)] += 1
                    continue

            if self.replace_type == "multi" and len(f_indices) > 1 and len(e_indices) == 1 and self.is_contiguous(f_indices):  # many-to-one
                e_index = e_indices[0]
                success, (new_src, new_tgt) = self.many_to_one_replace(src[f_indices[0]:f_indices[-1]+1], tgt[e_index])
                if success:
                    new_src_seq[f_indices[0]:f_indices[-1]+1] = [None] * len(f_indices)
                    new_src_seq[f_indices[0]] = new_src
                    new_tgt_seq[e_index] = new_tgt
                    orig_src_string = " ".join(src[f_indices[0]:f_indices[-1]+1])
                    if self.store_memory:
                        self.memory[Replacement(orig_src_string, tgt[e_index], new_src, new_tgt)] += 1

                    continue

            for index in f_indices:
                if self.backoff == "unk":
                    if src[index] not in self.src_voc:
                        new_src_seq[index] = "<@UNK>"

                elif self.backoff == "bpe":
                    new_src = self.apply_bpe_source(src[index])
                    assert self.allow_unk_character or "<unk>" not in new_src, "New source sequence %s contains <unk>" % new_src
                    new_src_seq[index] = new_src
                else:
                    raise NotImplementedError()

            for index in e_indices:
                if self.backoff == "unk":
                    if tgt[index] not in self.tgt_voc:
                        new_tgt_seq[index] = "<@UNK>"

                elif self.backoff == "bpe":
                    new_tgt = self.apply_bpe_target(tgt[index])
                    assert self.allow_unk_character or "<unk>" not in new_tgt, "New target sequence %s contains <unk>" % new_tgt
                    new_tgt_seq[index] = new_tgt
                else:
                    raise NotImplementedError()

            if self.is_contiguous(f_indices) and self.is_contiguous(e_indices):  # save in memory
                orig_src_string = " ".join(src[f_indices[0]:f_indices[-1]+1])
                orig_tgt_string = " ".join(tgt[e_indices[0]:e_indices[-1]+1])
                new_src_string = " ".join(new_src_seq[f_indices[0]:f_indices[-1]+1])
                new_tgt_string = " ".join(new_tgt_seq[e_indices[0]:e_indices[-1]+1])
                if self.store_memory:
                    self.memory[Replacement(orig_src_string, orig_tgt_string, new_src_string, new_tgt_string)] += 1

        new_src_seq = list(filter(None, new_src_seq))
        new_tgt_seq = list(filter(None, new_tgt_seq))
        return new_src_seq, new_tgt_seq

    def get_best_candidate(self, src: List[str], tgt: List[str], candidates: List[Candidate]) -> Tuple[str, str]:
        """
        candidates must be sorted by cos_sim, and then lex_prob in the descending order
        """
        best_pair = None
        best_score = float('-inf')
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

        assert best_pair is not None

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

    def replace_parallel_corpus(self, src_file: str, tgt_file: str, align_file: str, suffix: str, root_dir: str, print_per_lines: int=10000, first_n_lines: int=None):

        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        new_src_filename = os.path.join(root_dir, os.path.basename(src_file) + suffix)
        new_tgt_filename = os.path.join(root_dir, os.path.basename(tgt_file) + suffix)
        existing_files = []
        if os.path.exists(new_src_filename):
            existing_files.append(new_src_filename)

        if os.path.exists(new_tgt_filename):
            existing_files.append(new_tgt_filename)

        if len(existing_files) > 0:
            input("%s will be overwritten. Press Enter to proceed." % (" ".join(existing_files),))

        new_src_file = open(new_src_filename, 'w')
        new_tgt_file = open(new_tgt_filename, 'w')
        src_lines = open(src_file, 'r')
        tgt_lines = open(tgt_file, 'r')
        align_lines = open(align_file, 'r')

        for index, (src_line, tgt_line, align_line) in enumerate(zip_longest(src_lines, tgt_lines, align_lines)):

            if first_n_lines is not None and first_n_lines <= index:
                break

            if src_line is None or tgt_line is None or align_line is None:
                break

            src_tokens = src_line.strip().split(' ')
            tgt_tokens = tgt_line.strip().split(' ')
            align = Alignment.convert_string_to_alignment_dictionary(align_line)
            try:
                new_src_tokens, new_tgt_tokens = self.replace(src_tokens, tgt_tokens, align)
            except Exception as e:
                print("Failed at line %d." % (index + 1,), file=sys.stderr)
                logger.exception(e)
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
    def factory(cls, replace_type: str, store_memory: bool, src_w2v_model_path, tgt_w2v_model_path, src_w2v_model_topn,
                lex_e2f_path, lex_f2e_path, lex_topn, voc_path, src_w2v_lowercase=False,
                tgt_w2v_lowercase=False, bpe_vocab_path: str=None,
                backoff: str="unk", handle_numbers: bool=False):
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

        if backoff == "bpe":
            assert bpe_vocab_path is not None
            logger.info("Loading BPE vocab from %s" % bpe_vocab_path)
            with open(bpe_vocab_path, 'r') as f:
                bpe_vocab = json.load(f)
                src_bpe = BPE(bpe_vocab[0], use_separator=True)
                logger.info("Vocab size of source BPE: %d", len(src_bpe.get_vocab()))
                tgt_bpe = BPE(bpe_vocab[1], use_separator=True)
                logger.info("Vocab size of target BPE: %d", len(tgt_bpe.get_vocab()))
        elif backoff == "unk":
            src_bpe, tgt_bpe = None, None
        else:
            raise NotImplementedError()

        logger.info("Building Replacer instance")
        return cls(src_emb=src_emb, tgt_emb=tgt_emb, lex_e2f=lex_e2f,
                   lex_f2e=lex_f2e, src_voc=src_voc, tgt_voc=tgt_voc,
                   src_bpe=src_bpe, tgt_bpe=tgt_bpe, backoff=backoff,
                   replace_type=replace_type, store_memory=store_memory,
                   handle_numbers=handle_numbers)


def define_parser(parser):
    parser.add_argument('--src-w2v-model', required=True, type=str, help='Path to source word2vec model')
    parser.add_argument('--src-w2v-lowercase', action='store_true', help='Lowercase words before querying src word2vec')
    parser.add_argument('--tgt-w2v-model', required=True, type=str, help='Path to target word2vec model')
    parser.add_argument('--tgt-w2v-lowercase', action='store_true', help='Lowercase words before querying tgt word2vec')
    parser.add_argument('--train-first-n-lines', default=None, type=int, help='Process first %(metavar)s lines in training data', metavar='N')
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
    parser.add_argument('--replaced-suffix', default='', type=str,
                        help='Suffix for newly created training and dev data')
    parser.add_argument('--vocab', required=True, type=str, help='Path to vocabulary file')
    parser.add_argument('--bpe-vocab', default=None, type=str, help='Path to source BPE vocab')
    parser.add_argument('--memory', type=str, help='Save path to replacement memory', default=None)
    parser.add_argument('--backoff', choices=['bpe', 'unk'], default='unk', metavar='BACKOFF',
                        help='Use %(metavar)s for null aligned unknown words or unknown words in many-to-many links')
    parser.add_argument('--replace-type', required=True, choices=['1-to-1', 'multi'], 
                        help=textwrap.dedent(
                        '''
                        1-to-1: Replace unknown words in 1-to-1 links only
                        multi: Replace unknown words in 1-to-1, 1-to-many, many-to-1
                        '''))
    parser.add_argument('-n', '--handle-numbers', action='store_true')
    parser.add_argument('-r', '--root-dir', required=True, help='Path to save artifacts')
    # TODO: add src-sim, tgt-sim, prob, threshold
    # TODO: add embedding vocab option


def run(options):
    replacer = Replacer.factory(src_w2v_model_path=options.src_w2v_model,
                                src_w2v_lowercase=options.src_w2v_lowercase,
                                tgt_w2v_model_path=options.tgt_w2v_model,
                                tgt_w2v_lowercase=options.tgt_w2v_lowercase,
                                src_w2v_model_topn=options.src_w2v_model_topn,
                                lex_e2f_path=options.lex_e2f,
                                lex_f2e_path=options.lex_f2e,
                                lex_topn=options.lex_topn,
                                voc_path=options.vocab,
                                bpe_vocab_path=options.bpe_vocab,
                                backoff=options.backoff,
                                replace_type=options.replace_type,
                                store_memory=options.memory is not None,
                                handle_numbers=options.handle_numbers)

    if options.train_src is not None and options.train_tgt is not None and options.train_align is not None:
        logger.info("Processing training data")
        replacer.set_allow_unk_character(False)
        replacer.replace_parallel_corpus(options.train_src, options.train_tgt, options.train_align, options.replaced_suffix, options.root_dir, print_per_lines=10000, first_n_lines=options.train_first_n_lines)

    if options.dev_src is not None and options.dev_tgt is not None and options.dev_align is not None:
        logger.info("Processing dev data")
        replacer.set_allow_unk_character(True)
        replacer.replace_parallel_corpus(options.dev_src, options.dev_tgt, options.dev_align, options.replaced_suffix, options.root_dir, print_per_lines=100)

    if options.memory is not None:
        logger.info("Finally writing replacement memory")
        replacer.export_memory(options.memory)


def command_line(args=None):
    parser = argparse.ArgumentParser(description='Replace training data', formatter_class=argparse.RawTextHelpFormatter)
    define_parser(parser)
    options = parser.parse_args(args)
    run(options)

if __name__ == "__main__":
    command_line()
