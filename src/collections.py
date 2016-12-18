from typing import List, Any, Tuple
from collections import defaultdict


class UnionFind:
    def __init__(self, size):
        # A negative value denotes representative of set and the absolute value denotes the size of set
        # A positive value denotes next element.
        self.table = [-1 for _ in range(size)]

    # Get representative of a set
    def find(self, x):
        while self.table[x] >= 0:
            x = self.table[x]

        return x

    def union(self, x, y):
        s1 = self.find(x)
        s2 = self.find(y)
        if s1 != s2:
            if self.table[s1] >= self.table[s2]:
                # The smaller has more elements
                self.table[s1] += self.table[s2]
                self.table[s2] = s1
            else:
                self.table[s2] += self.table[s1]
                self.table[s1] = s2
            return True
        return False

    def get_groups(self):
        groups = []
        for i in range(len(self.table)):
            groups.append(self.find(i))

        return groups


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
