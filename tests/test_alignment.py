from unittest import TestCase
from unk_replacer.alignment import Alignment


class TestNumberHandler(TestCase):
    def test_get_index_mapping_1(self):
        s1 = "これ は 有用である と"
        s2 = "これ は 有用 である と"
        expected = [
            [0],
            [1],
            [2, 3],
            [4]
        ]
        mapping = Alignment.get_index_mapping(s1.split(' '), s2.split(' '))
        self.assertEqual(mapping, expected)

    def test_get_index_mapping_2(self):
        s1 = "これ は 有用である と"
        s2 = "これ は 有用 である という"

        with self.assertRaises(AssertionError):
            Alignment.get_index_mapping(s1.split(' '), s2.split(' '))

    def test_get_adjusted_alignment_1(self):
        sb = "これ は 有用である と ,"
        tb = "this is useful ,"
        align = "0-0 1-1 2-1 2-2 4-3"
        sa = "これ は 有用 である と ,"
        ta = "this is use ful ,"

        expected = "0-0 1-1 2-1 2-2 2-3 3-1 3-2 3-3 5-4"

        new_align = Alignment.get_adjusted_alignment(sb.split(' '), tb.split(' '), sa.split(' '), ta.split(' '), align)
        self.assertCountEqual(expected.split(' '), new_align.split(' '))

    def test_get_adjusted_alignment_2(self):
        sb = "これ は 有用である と ,"
        tb = "this is useful ,"
        align = "0-0 1-1 2-1 2-2 4-3"
        sa = "これ は 有用 である という ,"
        ta = "this is use ful ,"

        with self.assertRaises(AssertionError):
            Alignment.get_adjusted_alignment(sb.split(' '), tb.split(' '), sa.split(' '), ta.split(' '), align)
