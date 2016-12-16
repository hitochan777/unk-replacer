from unittest import TestCase
from src.collections import Trie


class TestTrie(TestCase):
    @classmethod
    def make_trie_example(cls):
        trie = Trie()
        trie.add(["a", "b", "c"], "hello", 1)
        trie.add(["a", "b", "c"], "foo", 2)
        trie.add(["a"], "foo", 1)
        trie.add(["b", "c", "d"], "hoho", 3)
        return trie

    def test_get_longest_match(self):
        trie = self.make_trie_example()
        trie.prune()
        match, index = trie.get_longest_match(["a", "b", "c", "d", "e"])
        self.assertEqual(match, "foo")
        self.assertEqual(index, 3)

    def test_get_longest_match_with_not_existing_seq(self):
        trie = self.make_trie_example()
        trie.prune()
        match, index = trie.get_longest_match(["not", "existing", "in", "the", "trie"])
        self.assertEqual(match, None)
        self.assertEqual(index, None)

    def test_get_longest_match_without_pruning(self):
        trie = self.make_trie_example()
        with self.assertRaises(AssertionError):
            _, _ = trie.get_longest_match(["a", "b", "c", "d", "e"])
