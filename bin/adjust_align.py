import argparse

from replacer.alignment import Alignment


def main(args=None):
    parser = argparse.ArgumentParser(description='Given two parallel corpora p1 and p2, where '
                                                 'some words in p2 are segmented into their sub-words, '
                                                 'and p1\'s word alignment, '
                                                 'return word alignment for p2')

    parser.add_argument('first-src', type=str, help='Path to p1 source file')
    parser.add_argument('first-tgt', type=str, help='Path to p1 source file')
    parser.add_argument('second-src', type=str, help='Path to p2 source file')
    parser.add_argument('second-tgt', type=str, help='Path to p2 source file')
    parser.add_argument('alignment', type=str, help='Path to word alignment file to p1')

    options = parser.parse_args(args)

    with open(options.first_src) as first_src, open(options.first_tgt) as first_tgt, \
        open(options.second_src) as second_src, open(options.second_tgt) as second_tgt, \
        open(options.alignment) as alignments:

        for sb, tb, sa, ta, align in zip(first_src, first_tgt, second_src, second_tgt, alignments):
            new_align_str = Alignment.get_adjusted_alignment(sb.split(' '), tb.split(' '), sa.split(' '), ta.split(' '), align)
            print(new_align_str)

if __name__ == "__main__":
    main()