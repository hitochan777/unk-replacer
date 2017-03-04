import argparse
import logging
import json
from typing import List, Tuple
from itertools import zip_longest
import re

from unk_replacer.lexical_dictionary import LexicalDictionary
from unk_replacer.number_normalizer import NumberHandler, NumberRestorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Restorer:
    def __init__(self, lex_e2f: LexicalDictionary, lex_f2e: LexicalDictionary, memory=None,
                 prob_threshold=0.1, attention_threshold=0.1, lex_backoff: bool=False, lex_top_n=None,
                 handle_numbers: bool=False, number_restorer: str="hankaku", no_unk_rep: bool=False):
        self.lex_e2f = lex_e2f
        self.lex_f2e = lex_f2e
        self.memory = memory
        self.prob_threshold = prob_threshold
        self.attention_threshold = attention_threshold

        if lex_backoff:
            logger.info("Lexical table backoff is True")

        self.lex_backoff = lex_backoff
        self.lex_top_n = lex_top_n

        self.count_back_substitute = 0
        self.count_changed_src = 0
        self.nb_no_dic_entry = 0

        self.print_every = 100
        
        self.handle_numbers = handle_numbers

        if self.handle_numbers:
            logger.info('number handling is ON')
        else:
            logger.info('number handling is OFF')

        if number_restorer == "hankaku":
            self.number_restorer = NumberRestorer.restore_to_hankaku
        elif number_restorer == "zenkaku":
            self.number_restorer = NumberRestorer.restore_to_zenkaku
        else:
            raise RuntimeError("Invalid number restorer")

        self.no_unk_rep = no_unk_rep
        if self.no_unk_rep:
            logger.info("Not replacing UNK symbols")

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
                translation = translation.strip()
                orig_input = orig_input.strip()
                replaced_input = replaced_input.strip()
                if len(translation) > 0:
                    translation_tokens = translation.split(' ')
                    orig_input_tokens = orig_input.split(' ')
                    replaced_input_tokens = replaced_input.split(' ')
                    if isinstance(attention, dict):
                        attention = attention["attn"]

                    restored_translation = self.restore(translation_tokens, orig_input_tokens, replaced_input_tokens, attention, log)
                    print(restored_translation, file=output)
                else:
                    print("" ,file=output)

                if (index + 1) % self.print_every == 0:
                    logger.info("Finished processing up to %d-th line" % (index + 1, ))

            self.print_statistics()

    def get_best_lexical_translation(self, f_word: str) -> Tuple[str, bool]:
        """
        Returns (best lexical translation of f_word, True) if there exist its translations;
        Otherwise returns (f_word, False)
        """
        candidates = self.lex_f2e.get_translations(
            cond=f_word,
            only_in_vocab=False,
            topn=self.lex_top_n
        )
        if len(candidates) == 0:
            return f_word, False  # copy source word because there is no translation

        best_e_word = None
        max_prob = 0.0
        for e_word, f2e_prob in candidates:
            e2f_prob = self.lex_e2f.get_prob(cond=e_word, word=f_word)
            cur_prob = (e2f_prob + f2e_prob)/2.0
            if cur_prob > max_prob:
                max_prob = cur_prob
                best_e_word = e_word

        return best_e_word, True

    def process_one_replaced_word(self, orig_src_seq: str, replaced_src_word: str, attention: List[float],
                                  translation: List[str], is_recovered: List[bool], recovered_translation: List[str], f_index: int):
        orig_src_seq_len = len(orig_src_seq.split(' '))
        max_score = float('-inf')  # type: float
        e_index_with_max_score = -1  # type: int
        for eIndex, attention_prob in enumerate(attention[:len(translation)]):
            if ("#T_UNK_%d#" % f_index) == translation[eIndex]:
                max_score = float('inf')  # any number > -inf suffices
                e_index_with_max_score = eIndex
                logger.info("replaced word %s was traslated to UNK symbol" % replaced_src_word)
                break

            if attention_prob < self.attention_threshold:
                continue

            # skip BPE vocab
            if translation[eIndex].endswith(("@@", "</w>")) or (eIndex > 0 and translation[eIndex - 1].endswith(("@@", "</w>"))):
                continue

            if self.handle_numbers and replaced_src_word in ["<@num:2d>","<@num:3d>","<@num:4d>", "<@num:big>"]:
                # numbers need to be aligned to be numbers
                if translation[eIndex] not in ["<@num:2d>","<@num:3d>","<@num:4d>", "<@num:big>"]:
                     continue
            else:
                # if a replaced source word is not a number related token, but the target word in focus is a number related token, continue to the next iteration
                if NumberHandler.is_number_token(translation[eIndex]) or (eIndex < len(translation) - 1 and NumberHandler.is_number_token(translation[eIndex + 1])):
                    continue

            if self.handle_numbers and NumberHandler.is_number_tag(replaced_src_word):
                score = attention_prob
            else:
                prob_e2f = self.lex_e2f.get_prob(cond=translation[eIndex], word=replaced_src_word)  # type: float
                prob_f2e = self.lex_f2e.get_prob(cond=replaced_src_word, word=translation[eIndex])  # type: float
                score = (prob_e2f + prob_f2e)/2.0 * attention_prob

            if score <= 0.0:
                continue

            if max_score < score:
                max_score = score
                e_index_with_max_score = eIndex

        if max_score > float('-inf'):  # (1)
            if is_recovered[e_index_with_max_score]:
                logger.warning("%d-th word is already recovered so skipping (1)" % e_index_with_max_score)
            else:
                is_recovered[e_index_with_max_score] = True
                self.count_back_substitute += 1
                if not (self.handle_numbers and NumberHandler.is_number_tag(replaced_src_word)):
                    if self.memory is not None:
                        if replaced_src_word in self.memory and translation[e_index_with_max_score] in self.memory[replaced_src_word]:
                            dic = self.memory[replaced_src_word][translation[e_index_with_max_score]]
                            if orig_src_seq in dic:
                                best_word = dic[orig_src_seq][0]  # choose the first one for now
                                recovered_translation[e_index_with_max_score] = best_word
                                logger.info("[1:memory] %s ➔ %s" % (translation[e_index_with_max_score], best_word))
                                assert type(best_word) == str, best_word
                                return

                        logger.info("%s not found in the memory" % orig_src_seq)

                    if orig_src_seq_len == 1:
                        best_word, in_dict = self.get_best_lexical_translation(orig_src_seq)
                        if in_dict:
                            logger.info("[1:dic] %s ➔ %s" % (translation[e_index_with_max_score], best_word))
                            recovered_translation[e_index_with_max_score] = best_word
                            return

                self.nb_no_dic_entry += 1
                if self.handle_numbers and replaced_src_word in ["<@num:2d>","<@num:3d>","<@num:4d>", "<@num:big>"]:
                    best_word = self.number_restorer(orig_src_seq)
                else:
                    best_word = orig_src_seq

                recovered_translation[e_index_with_max_score] = best_word
                logger.info("[1:copy] %s ➔ %s" % (translation[e_index_with_max_score], best_word))

                return

        if self.handle_numbers and replaced_src_word in ["<@num:2d>","<@num:3d>","<@num:4d>", "<@num:big>"]:
            pass
        elif self.memory is None:
            candidates_replaced_src = self.lex_f2e.get_translations(cond=replaced_src_word, only_in_vocab=False,
                                                                    prob_threshold=self.prob_threshold)
        else:
            if replaced_src_word in self.memory:
                candidates_replaced_src = list(self.memory[replaced_src_word].values())
                candidates_replaced_src = list(zip(candidates_replaced_src, [1.0] * len(candidates_replaced_src)))
            elif self.lex_backoff:
                candidates_replaced_src = self.lex_f2e.get_translations(
                    cond=replaced_src_word,
                    only_in_vocab=False,
                    prob_threshold=self.prob_threshold
                )
            else:
                logger.info("%s not in the replacement memory." % replaced_src_word)
                candidates_replaced_src = []

        if not (self.handle_numbers and replaced_src_word in ["<@num:2d>","<@num:3d>","<@num:4d>", "<@num:big>"]):
            for word, prob in candidates_replaced_src:
                assert prob >= self.prob_threshold
                indices = [
                    index for index, translation_word in enumerate(recovered_translation)
                    if translation_word == word and not is_recovered[index]
                    ]
                if len(indices) > 0:
                    if len(indices) > 1:
                        logger.warning("There are %d target words I can replace!!!"
                                       " For now I simply choose the first non-recovered one." % (len(indices),))

                    for index in indices:
                        if not is_recovered[index]:
                            best_index = indices[0]
                            break
                    else:
                        logger.warning("There were no non-recovered target indices for %s! Using next candidate..." % (word,))
                        continue

                    if self.memory is None:
                        assert orig_src_seq_len == 1
                        best_word, in_dict = self.get_best_lexical_translation(orig_src_seq)
                        if in_dict:
                            logger.info("[2:dict] %s ➔ %s" % (translation[best_index], best_word))
                            recovered_translation[best_index] = best_word
                            is_recovered[best_index] = True
                            self.count_back_substitute += 1
                            return
                    else:
                        if replaced_src_word in self.memory and translation[eIndex] in self.memory[replaced_src_word]:
                            dic = self.memory[replaced_src_word][translation[e_index_with_max_score]]
                            if orig_src_seq in dic:
                                best_word = dic[orig_src_seq][0]  # For now use the first one
                                assert isinstance(best_word, str), best_word
                                recovered_translation[best_index] = best_word
                                is_recovered[best_index] = True
                                self.count_back_substitute += 1
                                logger.info("[2:memory] %s ➔ %s" % (translation[best_index], best_word))
                                return

                        if self.lex_backoff and orig_src_seq_len == 1:
                            best_word, in_dict = self.get_best_lexical_translation(orig_src_seq)
                            if in_dict:
                                recovered_translation[best_index] = best_word
                                is_recovered[best_index] = True
                                self.count_back_substitute += 1
                                logger.info("[2:dict] %s ➔ %s" % (translation[best_index], best_word))
                                return

                    self.nb_no_dic_entry += 1
                    logger.info("[2:copy] %s ➔ %s" % (translation[best_index], orig_src_seq))
                    recovered_translation[best_index] = orig_src_seq
                    is_recovered[best_index] = True
                    self.count_back_substitute += 1
                    return
            else:
                logger.info("Not replaced")
        else:
            indices = [
                index for index, word in enumerate(recovered_translation)
                if word == replaced_src_word and not is_recovered[index]
            ]
            if len(indices) > 0:
                if len(indices) > 1:
                    logger.warning("There are %d target words I can replace!!!"
                                   " For now I simply choose the first non-recovered one." % (len(indices),))

                for index in indices:
                    if not is_recovered[index]:
                        best_index = indices[0]
                        break
                else:
                    logger.warning("There were no non-recovered target indices for %s!  Not replacing..." % (word,))
                    return

                self.nb_no_dic_entry += 1
                logger.info("[2:copy] %s ➔ %s" % (translation[best_index], self.number_restorer(orig_src_seq)))
                recovered_translation[best_index] = self.number_restorer(orig_src_seq)
                is_recovered[best_index] = True
                self.count_back_substitute += 1
                return
            else:
                logger.info("Not replaced")
                return

        return

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

        recovered_translation = list(translation)  # type: List[str]
        is_recovered = [False] * len(translation)  # type: List[bool]

        for idx_before, idx_after in log:
            assert isinstance(idx_before, list)
            assert isinstance(idx_after, list)

        for fIndices_before, fIndices_after in log:  # For each replaced source word
            assert len(fIndices_after) > 0

            if len(fIndices_after) != 1:
                continue

            orig_src_seq = ' '.join(orig_src[fIndices_before[0]:fIndices_before[-1] + 1])
            f_index = fIndices_after[0]
            if replaced_src[f_index] == "<@UNK>":
                logger.info("Skipping <@UNK>")
                continue

            if self.handle_numbers and replaced_src[f_index] in ["<@num:%d>" % i for i in range(13)]:
                continue

            self.count_changed_src += 1

            self.process_one_replaced_word(orig_src_seq=orig_src_seq, replaced_src_word=replaced_src[f_index],
                                           attention=attention[f_index], translation=translation,
                                           is_recovered=is_recovered,
                                           recovered_translation=recovered_translation, f_index=f_index)

        recovered_translation = self.restore_bpe(list(filter(None, recovered_translation)))
        if not self.no_unk_rep:
            recovered_translation = self.replace_unk_symbols(orig_src, recovered_translation, len(replaced_src), log)

        if self.handle_numbers:
            recovered_translation = NumberHandler.restore(recovered_translation)
            for index, token in enumerate(recovered_translation):
                if token in ["<@num:%d>" % i for i in range(13)]:
                    m = re.search(r'^<@num:(?P<num>\d+)>$', token)
                    assert m is not None
                    num = m.group("num")
                    recovered_translation[index] = self.number_restorer(num)

        return ' '.join(recovered_translation)

    def replace_unk_symbols(self, orig_src: List[str], translation: List[str], replaced_src_len: int, log) -> List[str]:
        output = list(translation)
        orig_idx_dic = dict()
        l1 = list(range(len(orig_src)))
        l2 = list(range(replaced_src_len))

        for idx_before, idx_after in log:
            if len(idx_before) == len(idx_after):  # temporary cure for the case when log contains <@UNK> replacement
                continue

            before_min, before_max = idx_before[0], idx_before[-1]
            assert before_min == min(idx_before)
            assert before_max == max(idx_before)
            after_min, after_max = idx_after[0], idx_after[-1]
            assert after_min == min(idx_after)
            assert after_max == max(idx_after)

            l1[before_min:before_max+1] = [None] * len(idx_before)
            l2[after_min:after_max+1] = [None] * len(idx_after)

        l1 = [x for x in l1 if x is not None]
        l2 = [x for x in l2 if x is not None]

        assert len(l1) == len(l2), (len(l1), len(l2))

        for i in range(len(l1)):
            orig_idx_dic[l2[i]] = l1[i]
        
        for index, word in enumerate(output):
            m = re.search(r"#T_UNK_(?P<f_index>\d+)#", word)
            if m is None:
                continue

            f_index = int(m.group("f_index"))
            if self.handle_numbers:
                if f_index not in orig_idx_dic:
                    logger.info("Not restoring because %s (%d-th word) is aligned to a number token" % (translation[index], index))
                    continue
            else:
                if f_index not in orig_idx_dic:
                    # a replaced word is translated to multiple UNK symbols, I believe there is no difference between recovering the multiple UNK symbols or leave them as they are.
                    continue

            orig_idx = orig_idx_dic[f_index]
            orig_word = orig_src[orig_idx]
            best_word, in_dict = self.get_best_lexical_translation(orig_word)
            if in_dict:
                output[index] = best_word
            else:
                output[index] = orig_word 

        return output

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
    def factory(cls, lex_e2f_path: str, lex_f2e_path: str, memory_path: str=None, lex_backoff: bool=False,
                lex_top_n=None, handle_numbers: bool=False, number_restorer: str="hankaku", no_unk_rep: bool=False):
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
                    memory[rep_src][rep_tgt][orig_src].append(orig_tgt)
        else:
            memory = None

        return cls(lex_e2f, lex_f2e, memory, lex_backoff=lex_backoff, lex_top_n=lex_top_n, handle_numbers=handle_numbers, number_restorer=number_restorer, no_unk_rep=no_unk_rep)


def define_parser(parser):
    parser.add_argument('--translation', required=True, type=str, help='Path to translation')
    parser.add_argument('--orig-input', required=True, type=str, help='Path to original input')
    parser.add_argument('--replaced-input', required=True, type=str, help='Path to replaced input')
    parser.add_argument('--output', required=True, type=str, help='Path to output file')
    parser.add_argument('--lex-e2f', required=True, type=str, help='Path to target to source lexical dictionary')
    parser.add_argument('--lex-f2e', required=True, type=str, help='Path to source to source lexical dictionary')
    parser.add_argument('--lex-top-n', type=float, default=None,
                        help='Consider top-n lexical translation candidates: default=%(default)s')
    parser.add_argument('--memory', default=None, type=str, help='Path to replacement memory')
    parser.add_argument('--replace-log', required=True, type=str, help='Path to replacement log')
    parser.add_argument('--attention', required=True, type=str, help='Path to attention')
    parser.add_argument('-b', '--lex-backoff', action='store_true',
                        help='Use lexical table when entry is not found in memory')
    parser.add_argument('-n', '--handle-numbers', action='store_true', help='If set, apply special handling to numbers')
    parser.add_argument('--number-restorer', default="hankaku", choices=["zenkaku", "hankaku"], help='Restorer for numbers')
    parser.add_argument('--no-unk-rep', action='store_true', help='If set, do not replace UNK symbols in the translations')


def run(options):
    # write out command line options to a file in JSON format
    option_log_path = options.output + ".restore.config.json"
    with open(option_log_path, "w") as option_log:
        json.dump(vars(options), option_log)

    replacer = Restorer.factory(lex_e2f_path=options.lex_e2f,
                                lex_f2e_path=options.lex_f2e,
                                memory_path=options.memory,
                                lex_backoff=options.lex_backoff,
                                lex_top_n=options.lex_top_n,
                                handle_numbers=options.handle_numbers,
                                number_restorer=options.number_restorer,
                                no_unk_rep=options.no_unk_rep)

    replacer.restore_file(options.translation, options.orig_input, options.replaced_input, options.output, options.attention, options.replace_log)


def command_line(args=None):
    parser = argparse.ArgumentParser(description='Restore the final translation', formatter_class=argparse.RawTextHelpFormatter)
    define_parser(parser)
    options = parser.parse_args(args)
    run(options)


if __name__ == "__main__":
    command_line()
