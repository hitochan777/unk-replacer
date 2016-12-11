from typing import Iterable, List, Dict
import argparse
import json

from src.bpe.apply_bpe import BPE
from src.word2vec import Word2Vec


class Replacer:

    def __init__(self, emb: Word2Vec, bpe: BPE, voc: Iterable[str],
                 memory: Dict[str, str], sim_threshold: float=0.3) -> None:

        self.emb = emb
        self.sim_threshold = sim_threshold
        self.bpe = bpe
        self.voc = voc
        assert isinstance(self.voc, Iterable[str]), type(self.voc)
        self.memory = memory

    def replace(self, src: List[str]):
        assert self.emb is not None

        new_words = []

        for index, word in enumerate(src):

            if word not in self.voc:
                best_fword = self.emb.most_similar_word(word)
                fwords.append(best_fword)
            else:
                fwords.append(fword)

        return fwords

    @classmethod
    def factory(cls, w2v_model_path: str, w2v_model_topn: str, memory: str,
                voc_path: str, w2v_lowercase: bool=False, bpe_code_path: str=None):
        with open(voc_path, 'r') as f:
            all_vocab = json.load(f)
            assert len(all_vocab) == 2
            voc = all_vocab[0]

        emb = Word2Vec(model_path=w2v_model_path, topn=w2v_model_topn,
                       lowercase_beforehand=w2v_lowercase)
        emb.set_vocab(voc)

        # TODO: prune memory

        if bpe_code_path is not None:
            with open(bpe_code_path, 'r') as f:
                bpe = BPE(f, use_separator=True)
        else:
            bpe = None

        return cls(emb=emb, voc=voc, bpe=bpe)


def main(args=None):
    parser = argparse.ArgumentParser(description='Replace training data')
    parser.add_argument('--w2v-model', required=True, type=str, help='Path to source word2vec model')
    parser.add_argument('--src-w2v-model-topn', metavar='K', default=10, type=int,
                        help='Use top %(metavar)s most similar words from source word2vec')
    parser.add_argument('--replaced_suffix', required=True, type=str,
                        help='Suffix for newly created training and dev data')
    parser.add_argument('--w2v-lowercase',
                        action='store_true', type=bool, help='Lowercase word before querying word2vec')
    parser.add_argument('--vocab', required=True, type=str, help='Path to vocabulary file')
    parser.add_argument('--bpe-code', default=None, type=str, help='Path to source BPE code')
    parser.add_argument('--memory', default=None, type=str, help='Path to replacement memory')
    # TODO: add src-sim threshold
    # TODO: add embedding vocab option

    options = parser.parse_args(args)

    replacer = Replacer.factory(w2v_model_path=options.src_w2v_model,
                                w2v_model_topn=options.src_w2v_model_topn,
                                w2v_lowercase=options.w2v_lowercase,
                                voc_path=options.vocab,
                                bpe_code_path=options.src_bpe_code,
                                memory=options.memory)

    if options.dev_src is not None and options.dev_tgt is not None and options.dev_align is not None:
        replacer.replace_parallel_corpus(options.dev_src, options.dev_tgt, options.dev_align, options.suffix)

if __name__ == "__main__":
    main()
