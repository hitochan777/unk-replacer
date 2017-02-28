from typing import Iterable, List, Tuple, Any
import argparse
import json
import os
import logging

from replacer.bpe.apply_bpe import BPE
from replacer.word2vec import Word2Vec
from replacer.collections import Trie
from replacer.number_normalizer import NumberHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Replacer:

    def __init__(self, emb: Word2Vec, bpe: BPE, voc: Iterable[str],
                 memory: Trie, sim_threshold: float=0.5, backoff: str="unk",
                 too_common_threshold=5000, mem_with_unk_only=True, force_word2vec_for_one_word: bool=False,
                 handle_numbers: bool=False) -> None:

        self.emb = emb  # type: Word2Vec
        self.sim_threshold = sim_threshold  # type: float
        self.bpe = bpe  # type: BPE
        self.voc = voc  # type: Iterable[str]
        assert isinstance(self.voc, list), type(self.voc)
        self.memory = memory  # type: Trie

        logger.info("Using %s as a backoff method" % backoff)
        self.backoff = backoff

        self.too_common_threshold = too_common_threshold
        self.mem_with_unk_only = mem_with_unk_only
        self.force_word2vec_for_one_word = force_word2vec_for_one_word

        self.handle_numbers = handle_numbers

    def get_replace_by_memory_lookup(self, seq: List[str]):
        start_idx = 0
        replacements = []
        while start_idx < len(seq):
            replaced_str, index = self.memory.get_longest_match(seq[start_idx:])
            if replaced_str is not None:
                assert index is not None
                src_words = seq[start_idx:start_idx+index]
                all_too_common = all(word in self.voc[:self.too_common_threshold] for word in src_words)

                if self.mem_with_unk_only:
                    all_in_vocab = all(word in self.voc for word in src_words)
                    if all_in_vocab:
                        start_idx += index
                        continue

                if not all_too_common and replaced_str != " ".join(src_words):
                    logger.info("[memory] %s→ %s" % (" ".join(src_words), replaced_str))
                    replacements.append(
                        (
                            (start_idx, start_idx + index),
                            replaced_str
                        )
                    )

                start_idx += index
            else:
                start_idx += 1

        return replacements

    def replace(self, seq: List[str]) -> Tuple[str, Any]:
        new_seq = list(seq)  # clone seq
        actions = []
        assert self.emb is not None

        if self.handle_numbers:
            for index, word in enumerate(seq):
                new_word =NumberHandler.process_number(word)  # note that process word returns the word itself if it contains no number related characters.
                if new_word != word:
                    actions.append(
                        (
                            (index, index + 1),
                            new_word
                        )
                    )

        # memory lookup
        if self.memory is not None:
            actions.extend(self.get_replace_by_memory_lookup(seq))

        for index, word in enumerate(seq):
            skip = False
            for action in actions:
                assert len(action) == 2
                start_index, end_index = action[0]
                if start_index <= index < end_index:
                    skip = True
                    break

            if skip:
                continue

            if word not in self.voc:
                most_similar_words = self.emb.most_similar_word(word)
                if len(most_similar_words) == 0 or most_similar_words[0].similarity < self.sim_threshold:
                    if self.backoff == "bpe":
                        new_seg = self.bpe.segment_word(word)
                    elif self.backoff == "unk":
                        new_seg = "<@UNK>"
                    else:
                        raise NotImplementedError()

                    actions.append(
                        (
                            (index, index + 1),
                            new_seg
                        )
                    )
                else:
                    actions.append(
                        (
                            (index, index + 1),
                            most_similar_words[0].word
                        )
                    )

        step = 0
        orig_seq = ["nc"] * len(seq)  # nc means not changed
        change_seq = [["nc"]] * len(seq)

        for action in actions:
            start_index, end_index = action[0]
            orig_seq[start_index:end_index] = ["#%d" % step] * (end_index - start_index)
            change_seq[start_index:end_index] = [None] * (end_index - start_index)
            change_seq[start_index] = ["#%d" % step] * len(action[1].split(' '))
            new_seq[start_index:end_index] = [None] * (end_index - start_index)
            new_seq[start_index] = action[1]
            step += 1
        else:
            # print(change_seq)
            change_seq = list(filter(None, change_seq))
            change_seq = [item for sublist in change_seq for item in sublist]

        replace_log = []
        for step_num in range(step):
            orig_indices = []
            change_indices = []
            for index, orig in enumerate(orig_seq):
                if orig == "#%d" % step_num:
                    orig_indices.append(index)

            for index, change in enumerate(change_seq):
                if change == "#%d" % step_num:
                    change_indices.append(index)

            replace_log.append((orig_indices, change_indices))

        logger.debug(replace_log)
        return ' '.join(filter(None, new_seq)), replace_log

    def replace_file(self, input_file, suffix, replace_log, root_dir):
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        output_file = os.path.join(root_dir, os.path.basename(input_file) + suffix)
        logs = []
        if os.path.isfile(output_file):
            input("%s will be overwritten. Press Enter to continue" % output_file)

        with open(input_file, 'r') as lines, open(output_file, 'w') as out:
            for index, line in enumerate(lines):
                print("Processing line %d" % (index + 1))
                new_line, log = self.replace(line.strip().split(' '))
                logs.append(log)
                print(new_line, file=out)

        with open(replace_log, 'w') as log_fs:
            logger.info("Writing replacement logs to %s" % replace_log)
            json.dump(logs, log_fs)

    @classmethod
    def build_memory(cls, memory_list: List, min_freq: int=0, force_word2vec_for_one_word: bool=False):
        trie = Trie()
        for memory in memory_list:
            assert isinstance(memory[1], int), memory  # replacement freq
            if memory[1] >= min_freq:
                if not force_word2vec_for_one_word or len(memory[0][0].split(' ')) > 1:
                    trie.add(memory[0][0].split(' '), memory[0][2], memory[1])

        trie.prune()
        return trie

    @classmethod
    def factory(cls, w2v_model_path: str, w2v_model_topn: str,
                voc_path: str, memory: str=None, w2v_lowercase: bool=False, 
                bpe_code_path: str=None, sim_threshold: float=0.5, emb_voc_size: int=10000,
                backoff: str="unk", too_common_threshold: int=5000,
                use_all_memory: bool=False, memory_min_freq: int=1,
                force_word2vec_for_one_word: bool=False, handle_numbers: bool=False):

        logger.info("Loading vocabulary")
        with open(voc_path, 'r') as f:
            all_vocab = json.load(f)
            assert len(all_vocab) == 2
            voc = all_vocab[0]

        logger.info("Lading word2vec model")
        emb = Word2Vec(model_path=w2v_model_path, topn=w2v_model_topn,
                       lowercase_beforehand=w2v_lowercase)
        emb.set_vocab(voc, topn=emb_voc_size)

        if memory is not None:
            logger.info("Building memory")
            with open(memory, 'r') as f:
                memory_list = json.load(f)
                memory = cls.build_memory(memory_list, memory_min_freq, force_word2vec_for_one_word)  # type: Trie

        if bpe_code_path is not None:
            logger.info("Loading BPE codes")
            with open(bpe_code_path, 'r') as f:
                bpe = BPE(f, use_separator=True)
        else:
            bpe = None
        
        logger.info("Building replacer")
        return cls(emb=emb, voc=voc, bpe=bpe, memory=memory, sim_threshold=sim_threshold, backoff=backoff, too_common_threshold=too_common_threshold, mem_with_unk_only=not use_all_memory, force_word2vec_for_one_word=force_word2vec_for_one_word, handle_numbers=handle_numbers)


def main(args=None):
    parser = argparse.ArgumentParser(description='Replace training data')
    parser.add_argument('--w2v-model', required=True, type=str, help='Path to source word2vec model')
    parser.add_argument('--w2v-model-topn', metavar='K', default=10, type=int,
                        help='Use top %(metavar)s most similar words from source word2vec')
    parser.add_argument('--input', required=True, type=str, help='Path to input file to replace')
    parser.add_argument('--replaced-suffix', required=True, type=str,
                        help='Suffix for newly created training and dev data')
    parser.add_argument('--w2v-lowercase',
                        action='store_true', help='Lowercase word before querying word2vec')
    parser.add_argument('--vocab', required=True, type=str, help='Path to vocabulary file')
    parser.add_argument('--bpe-code', default=None, type=str, help='Path to source BPE code')
    parser.add_argument('--memory', default=None, type=str, help='Path to replacement memory')
    parser.add_argument('--sim-threshold', default=0.5, type=float,
                        help='Threshold value for source cosine similarity')
    parser.add_argument('--emb-vocab-size', metavar='K',
                        type=int, default=10000, help='Use top %(metavar)s most frequent in-vocab words as replacement')
    parser.add_argument('--replace-log', type=str, required=True,
                        help='Path to log file that keeps track of which parts of input sentences were replaced.')
    parser.add_argument('--backoff', choices=['bpe', 'unk'], default='unk', metavar='BACKOFF',
                        help='Use %(metavar)s as a backoff')
    parser.add_argument('--too-common-threshold', type=int, default=5000, metavar="K",
                        help='If all the words in a original source phrase in the memory appear in top %(metavar)s most frequent words in the vocabulary, do not trust the memory and do not replace')
    parser.add_argument('--use-all-memory', action='store_true', help='If not set, replacement memories with at least one unknown words are used')
    parser.add_argument('--memory-min-freq', type=int, default=1, help='Minumum frequency threshold for the replacement memory. Default: %(default)s')
    parser.add_argument('--force-word2vec-for-one-word', action='store_true', help='If set, word2vec is used for one word even if replacement memory exits')
    parser.add_argument('-n', '--handle-numbers', action='store_true', help='If set, apply special handling to numbers')
    parser.add_argument('-r', '--root-dir', required=True, help='Path to save artifacts')

    options = parser.parse_args(args)

    # write out command line options to a file in JSON format
    option_log_path = options.input + options.replaced_suffix + ".test_replacer.config.json"
    with open(option_log_path, "w") as option_log:
        json.dump(vars(options), option_log)

    replacer = Replacer.factory(w2v_model_path=options.w2v_model,
                                w2v_model_topn=options.w2v_model_topn,
                                w2v_lowercase=options.w2v_lowercase,
                                voc_path=options.vocab,
                                bpe_code_path=options.bpe_code,
                                emb_voc_size=options.emb_vocab_size,
                                sim_threshold=options.sim_threshold,
                                memory=options.memory,
                                backoff=options.backoff,
                                too_common_threshold=options.too_common_threshold,
                                use_all_memory=options.use_all_memory,
                                memory_min_freq=options.memory_min_freq,
                                force_word2vec_for_one_word=options.force_word2vec_for_one_word,
                                handle_numbers=options.handle_numbers)

    replacer.replace_file(options.input, options.replaced_suffix, options.replace_log)

if __name__ == "__main__":
    main()
