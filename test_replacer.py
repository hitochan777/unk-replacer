from typing import Iterable, List, Tuple, Any
import argparse
import json
from os import path
from collections import defaultdict
import logging

from src.bpe.apply_bpe import BPE
from src.word2vec import Word2Vec

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Trie:
    def __init__(self):
        self.__final = False
        self.__children = defaultdict(lambda: Trie())
        self.__values = defaultdict(int)
        self.value = None

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
            max_val = 0
            max_key = None
            for key, value in node.__values.items():
                if value > max_val:
                    max_val = value
                    max_key = key

            node.value = max_key

        for key, child in node.__children.items():
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
            assert current.value is not None
            return current.value, index

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

    def replace_by_memory_lookup(self, seq: List[str]) -> List[str]:
        new_seq = []
        start_idx = 0
        while start_idx < len(seq):
            replaced_str, index = self.memory.get_longest_match(seq[start_idx:])
            if replaced_str is not None:
                assert index is not None
                print("%s is replaced by %s" % (" ".join(seq[start_idx:start_idx+index]), replaced_str))
                new_seq.append(replaced_str)
                start_idx += index
            else:
                new_seq.append(seq[start_idx])
                start_idx += 1

        return ' '.join(new_seq).split(' ')

    def replace(self, seq: List[str]) -> str:
        assert self.emb is not None

        # memory lookup
        if self.memory is not None:
            seq = self.replace_by_memory_lookup(seq)

        new_seq = [] 
        for index, word in enumerate(seq):
            if word.endswith(("@@", "</w>")):
                continue
            
            if word not in self.voc:
                most_similar_words = self.emb.most_similar_word(word)
                if len(most_similar_words) == 0 or most_similar_words[0].similarity < self.sim_threshold:
                    new_seg = self.bpe.segment_word(word)
                    new_seq.append(new_seg)
                else:
                    new_seq.append(most_similar_words[0].word)
            else:
                new_seq.append(word)

        return " ".join(new_seq)

    def replace_file(self, input_file, suffix):
        output_file = input_file + suffix
        if path.isfile(output_file):
            input("%s will be overwritten. Press Enter to continue" % output_file)

        with open(input_file, 'r') as lines, open(output_file, 'w') as out:
            for index, line in enumerate(lines):
                print("Processing line %d" % (index + 1))
                new_line = self.replace(line.strip().split(' '))
                print(new_line, file=out)

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
                bpe_code_path: str=None, sim_threshold: float=0.3, emb_voc_size: int=10000):

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
        return cls(emb=emb, voc=voc, bpe=bpe, memory=memory, sim_threshold=sim_threshold)


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

    options = parser.parse_args(args)

    replacer = Replacer.factory(w2v_model_path=options.w2v_model,
                                w2v_model_topn=options.w2v_model_topn,
                                w2v_lowercase=options.w2v_lowercase,
                                voc_path=options.vocab,
                                bpe_code_path=options.bpe_code,
                                emb_voc_size=options.emb_vocab_size,
                                sim_threshold=options.sim_threshold,
                                memory=options.memory)

    replacer.replace_file(options.input, options.replaced_suffix)

if __name__ == "__main__":
    main()
