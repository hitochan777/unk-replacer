import argparse
import logging
import json
from typing import List
from itertools import zip_longest

from src.lexical_dictionary import LexicalDictionary
from src.collections import Trie

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Restorer:
    def __init__(self, lex_e2f: LexicalDictionary, lex_f2e: LexicalDictionary, memory=None,
                 prob_threshold=0.1, attention_threshold=0.1, lex_backoff: bool=False):
        self.lex_e2f = lex_e2f
        self.lex_f2e = lex_f2e
        self.memory = memory
        self.prob_threshold = prob_threshold
        self.attention_threshold = attention_threshold

        if lex_backoff:
            logger.info("Lexical table backoff is True")

        self.lex_backoff = lex_backoff

        self.count_back_substitute = 0
        self.count_changed_src = 0
        self.nb_no_dic_entry = 0

        self.print_every = 100

    def print_statistics(self):
        print("Statistics:")
        print("%d/%d(%f) words were back-substituted" % (self.count_back_substitute, self.count_changed_src, 1.0*self.count_back_substitute/self.count_changed_src))
        print("%d words are copied because of no dictionary entry" % self.nb_no_dic_entry)

    def restore_file(self, translation_path: str, orig_input_path,
                     replaced_input_path: str, output_path: str, attention_path: str, log_path: str) -> None:
        with open(translation_path, 'r') as translations, open(orig_input_path, 'r') as orig_inputs, \
                open(replaced_input_path, 'r') as replaced_inputs, open(attention_path, 'r') as attention_fs, open(log_path, 'r') as logs_fs, open(output_path, 'w') as output:
            logs = json.load(logs_fs)
            attentions = json.load(attention_fs)
            for index, (translation, orig_input, replaced_input, attention, log) in enumerate(zip_longest(translations, orig_inputs, replaced_inputs, attentions, logs)):
                translation_tokens = translation.strip().split(' ')
                orig_input_tokens = orig_input.strip().split(' ')
                replaced_input_tokens = replaced_input.strip().split(' ')
                restored_translation = self.restore(translation_tokens, orig_input_tokens, replaced_input_tokens, attention, log)
                print(restored_translation, file=output)
                if (index + 1) % self.print_every == 0:
                    logger.info("Finished processing up to %d-th line" % (index + 1, ))

            self.print_statistics()

    def restore(self, translation: List[str], orig_src: List[str], replaced_src: List[str], attention, log):
        """
        (1) For each replaced source word, find a target word to which it has the highest attention
        among all whose attention is higher than self.attention_threshold
        (2) If there is no target word that satisfies (1) and the translation candidates for the replaced source
        word is in the translation, replace the target word only when it is not already replaced.
        (3) If (1) and (2) fails, no replacement because the translation for the replaced source words
        might be missing.
        """
        attention = list(map(list, zip_longest(*attention)))  # transpose attention
        assert self.lex_e2f is not None
        assert self.lex_f2e is not None
        assert attention is not None

        if len(attention) != len(replaced_src):
            assert len("".join(translation)) == 0, translation
            logger.debug("attention and the translation is empty. Returning empty list ")
            return []

        changed_indices = []  # type: List[int]
        recovered_translation = list(translation)  # type: List[str]
        is_recovered = [False] * len(translation)  # type: List[bool]

        for idx_before, idx_after in log:
            assert isinstance(idx_before, list)
            assert isinstance(idx_after, list)

        for fIndices_before, fIndices_after in log:  # For each replaced source word
            assert len(fIndices_after) > 0

            if len(fIndices_after) != 1:
                continue

            self.count_changed_src += 1

            fIndex = fIndices_after[0]
            max_score = float('-inf')  # type: float
            e_index_with_max_score = -1  # type: int
            for eIndex, attention_prob in enumerate(attention[fIndex]):
                if attention_prob < self.attention_threshold:
                    continue

                # skip BPE vocab
                if translation[eIndex].endswith(("@@", "</w>")) or (eIndex > 0 and translation[eIndex - 1].endswith(("@@", "</w>"))):
                    continue

                if self.memory is None:
                    prob_e2f = self.lex_e2f.get_prob(cond=replaced_src[fIndex], word=translation[eIndex])  # type: float
                    prob_f2e = self.lex_f2e.get_prob(cond=translation[eIndex], word=replaced_src[fIndex])  # type: float
                    score = (prob_e2f + prob_f2e)/2.0 * attention_prob
                else:
                    if replaced_src[fIndex] in self.memory and translation[eIndex] in self.memory[replaced_src[fIndex]]:
                        score = attention_prob
                    else:
                        continue

                if score <= 0.0:
                    continue

                if max_score < score:
                    max_score = score
                    e_index_with_max_score = eIndex

            if max_score > float('-inf'):  # (1)
                if is_recovered[e_index_with_max_score]:
                    logger.warning("%d-th word is already recovered so skipping (1)" % e_index_with_max_score)
                else:
                    if self.memory is not None:
                        if replaced_src[fIndex] in self.memory and translation[eIndex] in self.memory[replaced_src[fIndex]]:
                            dic = self.memory[replaced_src[fIndex]][translation[e_index_with_max_score]]
                            orig_src_seq = " ".join(orig_src[fIndices_before[0]:fIndices_before[-1] + 1])
                            if orig_src_seq in dic:
                                best_word = dic[orig_src_seq][0]  # choose the first one for now
                                assert type(best_word) == str, best_word

                            else:
                                best_word = None

                        else:
                            best_word = None

                    if best_word is None and len(fIndices_before) == 1:
                        candidates = self.lex_f2e.get_translations(cond=orig_src[fIndices_before[0]], only_in_vocab=False, topn=1)
                        if len(candidates) == 0:
                            best_word = orig_src[fIndices_before[0]]  # copy source word because there is no translation
                            self.nb_no_dic_entry += 1
                        else:
                            best_word = candidates[0].word

                    recovered_translation[e_index_with_max_score] = best_word
                    logger.info("[attn] %s ➔ %s" % (translation[e_index_with_max_score], best_word))
                    is_recovered[e_index_with_max_score] = True
                    self.count_back_substitute += 1
                    continue

            if self.memory is None:
                candidates_replaced_src = self.lex_f2e.get_translations(cond=replaced_src[fIndex], only_in_vocab=False,
                                                                       prob_threshold=self.prob_threshold)
            else:
                if replaced_src[fIndex] in self.memory:
                    candidates_replaced_src = list(self.memory[replaced_src[fIndex]].values())
                    candidates_replaced_src = list(zip(candidates_replaced_src, [1.0]* len(candidates_replaced_src)))
                elif self.lex_backoff:
                    candidates_replaced_src = self.lex_f2e.get_translations(cond=replaced_src[fIndex], only_in_vocab=False,
                                                                       prob_threshold=self.prob_threshold)
                else:
                    logger.info("%s not in the replacement memory." % replaced_src[fIndex])
                    candidates_replaced_src = []

            for word, prob in candidates_replaced_src:
                assert prob >= self.prob_threshold
                indices = [index for index, translation_word in enumerate(recovered_translation) if translation_word == word and not is_recovered[index]]
                if len(indices) > 0:
                    if len(indices) > 1:
                        logger.warning("There are %d target words I can replace!!! For now I simply choose the first non-recovered one." % (len(indices),))

                    for index in indices:
                        if not is_recovered[index]:
                            best_index = indices[0]
                            break
                    else:
                        logger.warning("There were no non-recovered target indices for %s! Using next candidate..." % (word,))
                        continue

                    orig_src_seq = " ".join(orig_src[fIndices_before[0]:fIndices_before[-1] + 1])
                    candidates_orig_src = None
                    if self.memory is None:
                        assert len(fIndices_before) == 1
                        candidates_orig_src = self.lex_f2e.get_translations(cond=orig_src[fIndices_before[0]], only_in_vocab=False, topn=1)
                    else:
                        if replaced_src[fIndex] in self.memory and translation[eIndex] in self.memory[replaced_src[fIndex]]:
                            dic = self.memory[replaced_src[fIndex]][translation[e_index_with_max_score]]
                            if orig_src_seq in dic:
                                candidates_orig_src = [dic[orig_src_seq][0]]  # For now use the first one
                                logger.info("Using memory to translate %s➔%s" % (orig_src_seq), candidates_orig_src[0])
                                assert type(best_word) == str, best_word
                                
                        if candidates_orig_src is None:
                            if self.lex_backoff and len(fIndices_before) == 1:
                                candidates_orig_src = self.lex_f2e.get_translations(cond=orig_src[fIndices_before[0]], only_in_vocab=False, topn=1)

                                if len(candidates_orig_src) > 0:
                                    logger.info("Backoff to lexical translation to translate %s➔%s" % (orig_src_seq, candidates_orig_src[0]))
                            else:
                                candidates_orig_src = []

                    if len(candidates_orig_src) == 0:
                        best_word = orig_src_seq
                        self.nb_no_dic_entry += 1
                        logger.info("[copy] %s ➔ %s" % (translation[best_index], best_word))
                    else:
                        best_word = candidates_orig_src[0][0]
                        logger.info("[dic] %s ➔ %s" % (translation[best_index], best_word))

                    recovered_translation[best_index] = best_word
                    is_recovered[best_index] = True
                    self.count_back_substitute += 1
                    break
            else:
                logger.info("Not replaced")

        # print(recovered_translation)
        recovered_translation = self.restore_bpe(list(filter(None, recovered_translation)))
        return ' '.join(recovered_translation)

    @staticmethod
    def restore_bpe(seq: List[str]) -> List[str]:
        result = []
        buf = []
        for word in seq:
            if word.endswith("@@"):
                buf.append(word[:-2])
            else:
                if word.endswith("</w>") and len(word[:-4]) != 0:
                    buf.append(word[:-4])
                else:  # word not in BPE voc
                    buf.append(word)

                result.append(''.join(buf))
                buf = []

        if len(buf) > 0:
            result.append(' '.join(buf))

        return result



    @classmethod
    def factory(cls, lex_e2f_path: str, lex_f2e_path: str, memory_path: str=None, lex_backoff: bool=False):
        logger.info("Loading e2f lexical dictionary from %s" % lex_e2f_path)
        lex_e2f = LexicalDictionary.read_lex_table(lex_e2f_path, topn=None)
        logger.info("Loading f2e lexical dictionary from %s" % lex_f2e_path)
        lex_f2e = LexicalDictionary.read_lex_table(lex_f2e_path, topn=None)

        if memory_path is not None:
            logger.info("Building memory")
            with open(memory_path, 'r') as f:
                memory_list = json.load(f)
                from collections import defaultdict
                memory = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
                for (orig_src, orig_tgt, rep_src, rep_tgt), freq in memory_list:
                    memory[rep_src][rep_tgt][orig_src].append(rep_tgt)

        return cls(lex_e2f, lex_f2e, memory, lex_backoff=lex_backoff)


def main(args=None):
    parser = argparse.ArgumentParser(description='Replace training data')
    parser.add_argument('--translation', required=True, type=str, help='Path to translation')
    parser.add_argument('--orig-input', required=True, type=str, help='Path to original input')
    parser.add_argument('--replaced-input', required=True, type=str, help='Path to replaced input')
    parser.add_argument('--output', required=True, type=str, help='Path to output file')
    parser.add_argument('--lex-e2f', required=True, type=str, help='Path to target to source lexical dictionary')
    parser.add_argument('--lex-f2e', required=True, type=str, help='Path to source to source lexical dictionary')
    parser.add_argument('--memory', default=None, type=str, help='Path to replacement memory')
    parser.add_argument('--replace-log', required=True, type=str, help='Path to replacement log')
    parser.add_argument('--attention', required=True, type=str, help='Path to attention')
    parser.add_argument('-b', '--lex-backoff', action='store_true', 
                        help='Use lexical table when entry is not found in memory')

    options = parser.parse_args(args)

    replacer = Restorer.factory(lex_e2f_path=options.lex_e2f,
                                lex_f2e_path=options.lex_f2e,
                                memory_path=options.memory,
                                lex_backoff=options.lex_backoff)

    replacer.restore_file(options.translation, options.orig_input, options.replaced_input, options.output, options.attention, options.replace_log)

if __name__ == "__main__":
    main()
