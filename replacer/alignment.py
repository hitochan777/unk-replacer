import logging
import itertools

from collections import defaultdict
import bisect

from typing import Iterator, Union, Dict, Tuple, List

from .collections import UnionFind

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Alignment(object):

    @classmethod
    def convert_string_to_alignment_dictionary(cls, line):
        dic = defaultdict(list)
        if line.rstrip() == "":
            return dic

        links = line.rstrip().split(" ")
        for link in links:
            fword, eword = list(map(int, link.split("-")))
            dic[fword].append(eword)

        return dic
        
    @classmethod
    def read_alignment(cls, filename):
        with open(filename, "r") as f:
            for line in f:
                dic = cls.convert_string_to_alignment_dictionary(line)
                yield dic

    @classmethod
    def enum_scc(cls, alignment: Union[Dict, str], src_len: int, tgt_len: int):
        """
        enumerate strongly connected component (scc) with additional information
        """

        uf = UnionFind(src_len + tgt_len)
        if isinstance(alignment, dict):
            def index_generator():
                for f_index, e_indices in alignment.items():
                    for e_index in e_indices:
                        yield (f_index, e_index)

        elif isinstance(alignment, str):
            if alignment.strip() == "":
                return []

            def index_generator():
                for link in alignment.strip().split(" "):
                    yield tuple(map(int, link.split("-")))

        else:
            raise NotImplementedError("Currently alignment can be of either str or dict type.")

        for f_index, e_index in index_generator():
            e_index += src_len
            uf.union(f_index, e_index)

        groups = uf.get_groups()
        f_groups = groups[:src_len]
        e_groups = groups[src_len:]
        group_dict = defaultdict(lambda: ([], []))  # type: Dict[int, Tuple[List[int], List[int]]]
        assert len(e_groups) == tgt_len
        for index, group in enumerate(f_groups):
            bisect.insort_left(group_dict[group][0], index)  # guarantee indices are in ascending order

        for index, group in enumerate(e_groups):
            bisect.insort_left(group_dict[group][1], index)  # guarantee indices are in ascending order

        scc = list(group_dict.values())

        return scc

    @classmethod
    def get_scc_with_unknowns(cls, alignment: Union[Dict, str], src: List[str],
                                    tgt: List[str], src_voc: Iterator[str], tgt_voc: Iterator[str]):

        scc = cls.enum_scc(alignment, len(src), len(tgt))
        filtered = []
        for f_indices, e_indices in scc:
            has_unknowns = False
            for index in f_indices:
                if src[index] not in src_voc:
                    has_unknowns = True
                    break

            if has_unknowns:
                filtered.append((f_indices, e_indices))
                continue

            for index in e_indices:
                if tgt[index] not in tgt_voc:
                    filtered.append((f_indices, e_indices))
                    break

        return filtered

    @staticmethod
    def get_index_mapping(l1: List[str], l2: List[str]):
        assert ''.join(l1) == ''.join(l2)
        assert len(l1) <= len(l2)
        p2 = 0
        mapping = []
        for p1 in range(len(l1)):
            indices = []
            s = ""
            while s != l1[p1]:
                s += l2[p2]
                indices.append(p2)
                p2 += 1

            mapping.append(indices)

        assert p2 == len(l2)
        assert len(mapping) == len(l1)
        return mapping

    @staticmethod
    def get_adjusted_alignment(orig_src: List[str], orig_tgt :List[str],
                               new_src: List[str], new_tgt: List[str], alignment: str):
        if alignment.strip() == '':
            return ''

        new_links = []

        assert ''.join(orig_src) == ''.join(new_src)
        assert ''.join(orig_tgt) == ''.join(new_tgt)

        map_src = Alignment.get_index_mapping(orig_src, new_src)
        map_tgt = Alignment.get_index_mapping(orig_tgt, new_tgt)

        for link_str in alignment.split(' '):
            f_index, e_index = list(map(int, link_str.split('-')))
            link_product = list(itertools.product(map_src[f_index], map_tgt[e_index]))
            new_links += link_product

        return ' '.join(map(lambda x: "%d-%d" % (x[0], x[1]), new_links))
