from typing import Iterable, List, Tuple, Any
import argparse
import json
from os import path
from collections import defaultdict

from src.bpe.apply_bpe import BPE
from src.word2vec import Word2Vec


class Trie:
    def __init__(self):
        self.__final = False
        self.__children = defaultdict(lambda: self.__class__())
        self.__values = defaultdict(int)

    def add(self, seq: List[Any], word: Any, value=1) -> None:
        current = self
        for unit in seq:
            current = current.__children[unit]

        current.__final = True
        current.__values[word] += value

    def prune(self):
        # Wrapper function for __prune(self, node)
        # Prune entries in each node that do not have the maximum frequency.
        # When there are multiple entries with the maximum frequency,
        # Only one entry is chosen randomly
        self.__prune(self)

    def __prune(self, node: 'Trie'):
        if node.__final:
            max_val = max(list(node.__values()))
            is_one_kept = False
            for key, value in node.__values:
                if is_one_kept or value < max_val:
                    del node.__values[key]
                else:
                    is_one_kept

            assert len(node.__values) == 1

        for _, child in self.__children.items():
            self.__prune(child)

    def get_longest_match(self, seq: List[str]) -> Tuple[str, int]:
        current = self
        index = 0
        for unit in seq:
            if unit in current.__children:
                current = current.__children[unit]
            else:
                break

            index += 1

        if current.__final:
            replace = list(current.__values.values())
            assert len(replace) == 0
            return replace[0], index

        return None, None


class Replacer:

    def __init__(self, emb: Word2Vec, bpe: BPE, voc: Iterable[str],
                 memory: Trie, sim_threshold: float=0.3) -> None:

        self.emb = emb  # type: Word2Vec
        self.sim_threshold = sim_threshold  # type: float
        self.bpe = bpe  # type: BPE
        self.voc = voc  # type: Iterable[str]
        assert isinstance(self.voc, Iterable[str]), type(self.voc)
        self.memory = memory  # type: Trie

    def replace_by_memory_lookup(self, seq: List[str]) -> str:
        new_seq = []
        start_idx = 0
        while start_idx < len(seq):
            replaced_str, index = self.memory.get_longest_match(seq[start_idx:])
            if replaced_str is not None:
                assert index is not None
                new_seq.append(replaced_str)

            start_idx = index

        return ' '.join(new_seq)

    def replace(self, seq: List[str]) -> str:
        assert self.emb is not None

        # memory lookup
        if self.memory is not None:
            seq_after_memory_lookup = self.replace_by_memory_lookup(seq)

        new_seq = list(seq_after_memory_lookup)  # clone a list
        for index, word in enumerate(seq_after_memory_lookup):
            if word not in self.voc:
                most_similar_word = self.emb.most_similar_word(word)[0]
                if most_similar_word.similarity < self.sim_threshold:
                    new_seg = self.bpe.segment_word(word)
                    new_seq.append(new_seg)
                else:
                    new_seq.append(most_similar_word.word)
            else:
                new_seq.append(word)

        return " ".join(new_seq)

    def replace_file(self, input_file, suffix):
        output_file = input_file + suffix
        if path.isfile(output_file):
            input("%s will be overwritten. Press Enter to continue" % output_file)

        with open(input_file, 'r') as lines, open(output_file, 'r') as out:
            for line in lines:
                new_seq = self.replace(line.strip().split(' '))
                new_line = " ".join(new_seq)
                print(new_line, file=out)

    @classmethod
    def build_memory(cls, memory_list):
        trie = Trie()
        for memory in memory_list:
            assert isinstance(memory[1], int)
            trie.add(memory[0][0].split(' '), memory[0][2], memory[1])

        trie.prune()
        return trie

    @classmethod
    def factory(cls, w2v_model_path: str, w2v_model_topn: str, memory: str,
                voc_path: str, w2v_lowercase: bool=False, bpe_code_path: str=None,
                sim_threshold: float=0.3, emb_voc_size: int=10000):
        with open(voc_path, 'r') as f:
            all_vocab = json.load(f)
            assert len(all_vocab) == 2
            voc = all_vocab[0]

        emb = Word2Vec(model_path=w2v_model_path, topn=w2v_model_topn,
                       lowercase_beforehand=w2v_lowercase)
        emb.set_vocab(voc, topn=emb_voc_size)

        structured_memory = cls.build_memory(memory)  # type: Trie

        if bpe_code_path is not None:
            with open(bpe_code_path, 'r') as f:
                bpe = BPE(f, use_separator=True)
        else:
            bpe = None

        return cls(emb=emb, voc=voc, bpe=bpe, memory=structured_memory, sim_threshold=sim_threshold)


def main(args=None):
    parser = argparse.ArgumentParser(description='Replace training data')
    parser.add_argument('--w2v-model', required=True, type=str, help='Path to source word2vec model')
    parser.add_argument('--src-w2v-model-topn', metavar='K', default=10, type=int,
                        help='Use top %(metavar)s most similar words from source word2vec')
    parser.add_argument('--input', required=True, type=str, help='Path to input file to replace')
    parser.add_argument('--replaced_suffix', required=True, type=str,
                        help='Suffix for newly created training and dev data')
    parser.add_argument('--w2v-lowercase',
                        action='store_true', help='Lowercase word before querying word2vec')
    parser.add_argument('--vocab', required=True, type=str, help='Path to vocabulary file')
    parser.add_argument('--bpe-code', default=None, type=str, help='Path to source BPE code')
    parser.add_argument('--memory', default=None, type=str, help='Path to replacement memory')
    parser.add_argument('--sim-threshold', default=0.3, type=str,
                        help='Threshold value for source cosine similarity')
    parser.add_argument('--emb-vocab-size', metavar='K',
                        type=int, default=10000, help='Use top %(metavar)s most frequent in-vocab words as replacement')

    options = parser.parse_args(args)

    replacer = Replacer.factory(w2v_model_path=options.src_w2v_model,
                                w2v_model_topn=options.src_w2v_model_topn,
                                w2v_lowercase=options.w2v_lowercase,
                                voc_path=options.vocab,
                                bpe_code_path=options.src_bpe_code,
                                emb_vocab_size=options.emb_vocab_size,
                                sim_threshold=options.sim_threshold,
                                memory=options.memory)

    replacer.replace_file(options.input, options.suffix)

if __name__ == "__main__":
    main()
