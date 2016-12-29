from typing import Iterable, List, Tuple, Any
import argparse
import json
from os import path
import logging

from src.bpe.apply_bpe import BPE
from src.word2vec import Word2Vec
from src.collections import Trie

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Replacer:

    def __init__(self, emb: Word2Vec, bpe: BPE, voc: Iterable[str],
                 memory: Trie, sim_threshold: float=0.3, backoff: str="unk") -> None:

        self.emb = emb  # type: Word2Vec
        self.sim_threshold = sim_threshold  # type: float
        self.bpe = bpe  # type: BPE
        self.voc = voc  # type: Iterable[str]
        assert isinstance(self.voc, Iterable[str]), type(self.voc)
        self.memory = memory  # type: Trie

        logger.info("Using %s as a backoff method" % backoff)
        self.backoff = backoff

    def get_replace_by_memory_lookup(self, seq: List[str]):
        new_seq = []
        start_idx = 0
        replacements = []
        while start_idx < len(seq):
            replaced_str, index = self.memory.get_longest_match(seq[start_idx:])
            if replaced_str is not None:
                assert index is not None
                print("%s is replaced by %s" % (" ".join(seq[start_idx:start_idx+index]), replaced_str))
                replacements.append(
                    (
                        (start_idx, start_idx + index),
                        replaced_str
                    )
                )
                start_idx += index
            else:
                new_seq.append(seq[start_idx])
                start_idx += 1

        return replacements

    def replace(self, seq: List[str]) -> Tuple[str, Any]:
        new_seq = list(seq)  # clone seq
        actions = []
        assert self.emb is not None

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

    def replace_file(self, input_file, suffix, replace_log):
        output_file = input_file + suffix
        logs = []
        if path.isfile(output_file):
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
    def build_memory(cls, memory_list: List):
        trie = Trie()
        for memory in memory_list:
            assert isinstance(memory[1], int), memory
            if memory[0][0] == memory[0][2]:
                continue

            trie.add(memory[0][0].split(' '), memory[0][2], memory[1])

        trie.prune()
        return trie

    @classmethod
    def factory(cls, w2v_model_path: str, w2v_model_topn: str,
                voc_path: str, memory: str=None, w2v_lowercase: bool=False, 
                bpe_code_path: str=None, sim_threshold: float=0.3, emb_voc_size: int=10000,
                backoff: str="unk"):

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
                memory = cls.build_memory(memory_list)  # type: Trie

        if bpe_code_path is not None:
            logger.info("Loading BPE codes")
            with open(bpe_code_path, 'r') as f:
                bpe = BPE(f, use_separator=True)
        else:
            bpe = None
        
        logger.info("Building replacer")
        return cls(emb=emb, voc=voc, bpe=bpe, memory=memory, sim_threshold=sim_threshold, backoff=backoff)


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
    parser.add_argument('--sim-threshold', default=0.6, type=str,
                        help='Threshold value for source cosine similarity')
    parser.add_argument('--emb-vocab-size', metavar='K',
                        type=int, default=10000, help='Use top %(metavar)s most frequent in-vocab words as replacement')
    parser.add_argument('--replace-log', type=str, required=True,
                        help='Path to log file that keeps track of which parts of input sentences were replaced.')
    parser.add_argument('--backoff', choices=['bpe', 'unk'], default='unk', metavar='BACKOFF',
                        help='Use %(metavar)s as a backoff')

    options = parser.parse_args(args)

    replacer = Replacer.factory(w2v_model_path=options.w2v_model,
                                w2v_model_topn=options.w2v_model_topn,
                                w2v_lowercase=options.w2v_lowercase,
                                voc_path=options.vocab,
                                bpe_code_path=options.bpe_code,
                                emb_voc_size=options.emb_vocab_size,
                                sim_threshold=options.sim_threshold,
                                memory=options.memory,
                                backoff=options.backoff)

    replacer.replace_file(options.input, options.replaced_suffix, options.replace_log)

if __name__ == "__main__":
    main()
